
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils import data
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torchvision
from torchvision import transforms
from models.decoderlstm import AttentionGru
from train_attention_gru import CaptionAttentionGru
from models.encoder import EncoderCNN
from bert_text_classifier import BertClassifer
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import copy
from data_loader import get_dataset, get_styled_dataset, flickr_collate_fn, ConcatDataset, FlickrCombiner, flickr_collate_style
import build_vocab
from build_vocab import Vocab
import pickle
import random
from pytorch_lightning.loggers import WandbLogger   
from utils import set_all_parameters, flip_parameters_to_tensors, WordVectorLoader, cap_to_text, cap_to_text_gt, metric_score, metric_score_test
from pytorch_lightning.callbacks import ModelCheckpoint
import datasets
import numpy as np
import random 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class HyperNet(pl.LightningModule):
    def __init__(self, feature_size, embed_size, hidden_size, vocab_size, vocab, num_layers=1, lr=1e-6, mixup=False, alpha=0.3):
        super().__init__()

        self.hparams['feature_size'] = feature_size
        self.hparams['vocab_size'] = vocab_size
        self.hparams['embed_size'] = embed_size
        self.hparams['hidden_size'] = hidden_size
        self.vocab = vocab
        self.hparams['lr'] = lr
        self.hparams['num_layers'] = num_layers
        self.teacher_forcing_proba = 0.0
        self.beam_size = 3
        self.metrics = ['bleu', 'meteor', 'rouge']
        self.mixup = mixup
        self.metrics = [datasets.load_metric(name) for name in self.metrics]
        self.alpha = alpha

        # create image encoder
        self.image_encoder = EncoderCNN()
            
        self.captioner = AttentionGru(2048, feature_size, embed_size, hidden_size, vocab_size, num_layers=num_layers, p=0.0)
        # create HN base
        N = 1
        M = 500
        self.hn_base = nn.Sequential(
            nn.Linear(embed_size, N*embed_size),
            nn.LeakyReLU(),
            nn.Linear(N*embed_size, N*embed_size),
            nn.LeakyReLU()
        )
        self.hn_heads = []
        for name, W in self.captioner.gru.named_parameters():
            if name=='embed.weight':
                continue
            if name=='fc_out.weight':
                continue
            if name=='fc_out.bias':
                continue

            w_size = len(W.flatten())
            if w_size < N*embed_size:
                self.hn_heads.append(nn.Sequential(
                    nn.Linear(N*embed_size, N),
                    nn.LeakyReLU(),
                    nn.Linear(w_size, w_size)
                ))
            else:
                if w_size // M < N*embed_size:
                    self.hn_heads.append(nn.Sequential(
                        nn.Linear(N*embed_size, N*embed_size),
                        nn.LeakyReLU(),
                        nn.Linear(N*embed_size, w_size)
                    ))
                else:
                    self.hn_heads.append(nn.Sequential(
                        nn.Linear(N*embed_size, w_size//M),
                        nn.LeakyReLU(),
                        nn.Linear(w_size//M, w_size)
                    ))
            

        self.hn_heads = nn.ModuleList(self.hn_heads)


    def load_glove_emb(self, emb_path):
        loader = WordVectorLoader(self.hparams['embed_size'])
        loader.load_glove(emb_path)
        emb_mat = loader.generate_embedding_matrix(self.vocab.w2i,
                                                   self.vocab.ix-1, 'norm')
        emb_mat = torch.FloatTensor(emb_mat)
        self.captioner.embed = self.captioner.embed.from_pretrained(emb_mat,
                                                                    freeze=False)
    
    def forward(self, x):
        base_feat = self.hn_base(x)

        heads_out = []
        for head in self.hn_heads:
            heads_out.append(head(base_feat).flatten())
        
        heads_out = torch.cat(heads_out, dim=0)
        flip_parameters_to_tensors(self.captioner.gru)
        set_all_parameters(self.captioner.gru, heads_out.reshape(1,-1))
        return self.captioner

    def configure_optimizers(self):
        params = list(self.hn_heads.parameters())
        params.extend(list(self.hn_base.parameters()))
        params.extend(list(self.captioner.feature_fc.parameters()))
        params.extend(list(self.captioner.embed.parameters()))
        params.extend(list(self.captioner.fc.parameters()))
        params.extend(list(self.captioner.attention.parameters()))
        params.extend(list(self.captioner.init_h.parameters()))
        optimizer = torch.optim.Adam(params, lr=self.hparams['lr'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, cooldown=2, factor=0.5)
        return [optimizer], [{'scheduler': scheduler, 'monitor': 'val_loss with TF', 'interval': 'epoch'}]


    def training_step(self, train_batch, batch_idx):
  
        imgs, (style, (caps, lengths)) = train_batch
        style = torch.tensor([self.vocab(style)])
        style = style.type(torch.LongTensor)
        style = style.to(self.device)
        style_embed = self.captioner.embed(style)

        self.forward(style_embed)
        img_feats = self.image_encoder(imgs.float())
        caps_pred, _ = self.captioner(img_feats, caps.long(), self.teacher_forcing_proba)
        if self.mixup:
            style_list = ["factual", "humour", "romantic"]
            coeff = random.random()
            no_style = random.choice(style_list)
            style_list.remove(no_style)
            style1, style2 = style_list[0], style_list[1]
            label1, label2= torch.tensor([style_classifier.labels[style1]]).to(device), torch.tensor([style_classifier.labels[style2]]).to(device)
            style1 = torch.tensor([self.vocab(style1)])
            style1 = style1.type(torch.LongTensor)
            style1 = style1.to(self.device)
            style2 = torch.tensor([self.vocab(style2)])
            style2 = style2.type(torch.LongTensor)
            style2 = style2.to(self.device)
            style_embed1 = self.captioner.embed(style1)
            style_embed2 = self.captioner.embed(style2)
            mixup_style = coeff*style_embed1 + (1-coeff)*style_embed2 
            self.forward(mixup_style)
            caps_pred_mixup, _ = self.captioner(img_feats, caps.long(), 1.0)
            for i in range(len(caps)):
                gt_idx = torch.squeeze(caps_pred_mixup[i])
                text = cap_to_text(gt_idx, self.vocab, tokenized=False)
            caption = style_classifier.tokenizer(text, 
                                padding='max_length', max_length = 25, truncation=True,
                                    return_tensors="pt") 
            mask = caption['attention_mask'].to(device)
            input_id = caption['input_ids'].squeeze(1).to(device)
            style_pred = style_classifier(input_id, mask)            
            if no_style == "factual":
                target = torch.tensor([0.0, coeff, 1-coeff]).to(device)
            elif no_style == "humour":
                target = torch.tensor([coeff, 0.0, 1-coeff]).to(device)
            else:
                target = torch.tensor([coeff,1-coeff, 0.0]).to(device)
            style_loss = F.mse_loss(style_pred, target)  


        loss =  F.cross_entropy(caps_pred.view(-1, self.hparams['vocab_size']), caps.view(-1).long(), ignore_index=self.vocab.w2i['<pad>']) 
        #l2_lambda = 0.00000001
        #l2_reg = torch.tensor(0.).to(device)
        #for param in self.parameters():
        #    l2_reg += torch.norm(param)
        #loss += l2_lambda * l2_reg         
        loss_ce = loss
        if self.mixup:
            loss =  self.alpha * loss + (1-self.alpha) * style_loss
            self.log('style loss', style_loss)
            self.log('cross entropy loss', loss_ce)

        bleu1, bleu2, bleu3, bleu4, meteor, rouge = metric_score(caps, caps_pred, self.vocab, self.metrics)
        self.log('meteor train', meteor)
        self.log('bleu 1 train', bleu1)
        self.log('bleu 2 train', bleu2)
        self.log('bleu 3 train', bleu3)
        self.log('bleu 4 train', bleu4)
        self.log('rouge train', rouge)
        self.log('train_loss', loss)
        
        self.log('LR', self.hparams['lr'])
        #if self.teacher_forcing_proba < 0.5: 
        #    self.teacher_forcing_proba = self.teacher_forcing_proba * 1.0005
        return loss

    def validation_step(self, val_batch, batch_idx):

        imgs, (style, (caps, lengths)) = val_batch
        style = torch.tensor([vocab(style)])
        style = style.type(torch.LongTensor)
        style = style.to(self.device)
        style_embed = self.captioner.embed(style)

        captioner = self.forward(style_embed)
        img_feats = self.image_encoder(imgs.float())

        caps_pred_teacher_forcing, _ = captioner(img_feats, caps.long(), self.teacher_forcing_proba)
        caps_pred, _ = captioner(img_feats, caps.long(), 1.0)

        # caps = nn.functional.one_hot(caps.to(torch.int64), num_classes=self.voc_size).float()
        loss = F.cross_entropy(caps_pred.view(-1, self.hparams['vocab_size']),
                               caps.view(-1).long(), ignore_index=self.vocab.w2i['<pad>'])
        
        bleu1, bleu2, bleu3, bleu4, meteor, rouge = metric_score(caps, caps_pred, self.vocab, self.metrics)
        
        self.log('val_loss', loss, sync_dist=True)
        self.log('meteor', meteor)
        self.log('bleu 1', bleu1)
        self.log('bleu 2', bleu2)
        self.log('bleu 3', bleu3)
        self.log('bleu 4', bleu4)
        self.log('rouge', rouge)


        loss_tf = F.cross_entropy(caps_pred_teacher_forcing.view(-1, self.hparams['vocab_size']),
                               caps.view(-1).long(), ignore_index=self.vocab.w2i['<pad>'])
        self.log('val_loss with TF', loss_tf, sync_dist=True)
        
    def test_step(self, test_batch, batch_idx):
        imgs, (style, (caps, lengths)) = test_batch
        style = torch.tensor([self.vocab(style)])
        style = style.type(torch.LongTensor)
        style = style.to(self.device)
        style_embed = self.captioner.embed(style)
        self.captioner = self.forward(style_embed)
        
        features = self.image_encoder(imgs.float())
        encoder_out = self.captioner.feature_fc(features)
        encoder_dim = encoder_out.size(-1)
        encoder_out = encoder_out.view(1, -1, encoder_dim) 
        num_pixels = encoder_out.size(1)
        k = self.beam_size
        vocab_size = self.hparams['vocab_size']
        encoder_out = encoder_out.expand(k, num_pixels, encoder_dim) 
        k_prev_words = torch.LongTensor([[0]] * k).to(device)
        seqs = k_prev_words
        top_k_scores = torch.zeros(k, 1).to(device)
        complete_seqs = []
        complete_seqs_scores = []
        step = 1
        h = self.captioner.init_hidden(encoder_out)
        while True:
            embeddings = self.captioner.embed(k_prev_words).squeeze(1)  
            if k_prev_words[0][0] == 0:
                embeddings[:][:][:][:][:][:] = 0
            context, atten_weight= self.captioner.attention(encoder_out, h) 
            input_concat = torch.cat([embeddings, context], 1)
            h = self.captioner.gru(input_concat, h)  
            scores = self.captioner.fc(h)  
            scores = F.log_softmax(scores, dim=1)
            # top_k_scores: [s, 1]
            scores = top_k_scores.expand_as(scores) + scores  # [s, vocab_size]
            # For the first step, all k points will have the same scores (since same k previous words, h, c)
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
            else:
                # Unroll and find top scores, and their unrolled indices
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

            # Convert unrolled indices to actual indices of scores
            prev_word_inds = top_k_words // vocab_size  # (s)
            next_word_inds = top_k_words % vocab_size  # (s)
            # Add new words to sequences
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                                next_word != self.vocab.w2i['</s>']]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))
            # Set aside complete sequences
            if len(complete_inds) > 0:
                Caption_End = True
                compute = True
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly
            # Proceed with incomplete sequences
            if k == 0:
                break

            seqs = seqs[incomplete_inds]
            h = h[prev_word_inds[incomplete_inds]]
            encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)
            # Break if things have been going on too long
            if step > 50:
                Caption_End = True
                compute = False
                break
            step += 1

        # choose the caption which has the best_score.
        Caption_End = True
        assert Caption_End
        if compute:
            indices = complete_seqs_scores.index(max(complete_seqs_scores))
            caps_pred_beam = complete_seqs[indices]




            bleu1_beam, bleu2_beam, bleu3_beam, bleu4_beam, meteor_beam, rouge_beam = metric_score_test(caps, torch.tensor(caps_pred_beam), self.vocab, self.metrics)
            self.log('meteor beam', meteor_beam)
            self.log('bleu 1 beam', bleu1_beam)
            self.log('bleu 2 beam', bleu2_beam)
            self.log('bleu 3 beam', bleu3_beam)
            self.log('bleu 4 beam', bleu4_beam)
            self.log('rouge beam', rouge_beam)

        
        img_feats = self.image_encoder(imgs.float())
        caps_pred, _ = self.captioner(img_feats, caps.long(), 1.0)
        bleu1, bleu2, bleu3, bleu4,  meteor, rouge = metric_score(caps, caps_pred, self.vocab, self.metrics)
        self.log('meteor', meteor)
        self.log('bleu 1', bleu1)
        self.log('bleu 2', bleu2)
        self.log('bleu 3', bleu3)
        self.log('bleu 4', bleu4)
        self.log('rouge', rouge)
              
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('HyperNet')
        parser.add_argument("--embed_size", type=int, default=200)
        parser.add_argument("--hidden_size", type=int, default=150)
        parser.add_argument("--num_layers", type=int, default=2)
        parser.add_argument("--type", type=str, default='gru')
        parser.add_argument("--lr", type=float, default=5e-3)
        return parent_parser

if __name__ == "__main__":
    img_path = "data/flickr7k_images"
    cap_path = "data/factual_train.txt"
    cap_path_humor = "data/humor/funny_train.txt"
    cap_path_romantic = "data/romantic/romantic_train.txt"
    glove_path = "/cortex/users/cohenza4/glove.6B.200d.txt"
    path_hn_factual= "/cortex/users/cohenza4/checkpoint_gru/hn/hn_factual.ckpt"
    path_hn_humour= "/cortex/users/cohenza4/checkpoint_gru/hn/hn_humour.pt"
    path_hn_romantic= "/cortex/users/cohenza4/checkpoint_gru/hn/hn_romantic.pt"
    path_all = "/cortex/users/cohenza4/checkpoint_gru/hn/hn_all.pt"
    gru_path = "/cortex/users/cohenza4/checkpoint_gru/factual/epoch=18-step=1584.ckpt"
    style_classifier_path = "/cortex/users/cohenza4/checkpoint/style_classifier/epoch=8-step=1574.ckpt"
    save_path = "/cortex/users/cohenza4/checkpoint/HN/romantic/"

    # data
    with open("data/vocab.pkl", 'rb') as f:
        vocab = pickle.load(f)
    orig_dataset = get_dataset(img_path, cap_path, vocab)
    humor_dataset = get_styled_dataset(cap_path_humor, vocab)
    romantic_dataset = get_styled_dataset(cap_path_romantic, vocab)


    

    data_concat = ConcatDataset(orig_dataset, humor_dataset, romantic_dataset)
    '''
    lengths = [int(len(data_concat)*0.8),
            len(data_concat) - int(len(data_concat)*0.8)]
    train_data, val_data = torch.utils.data.random_split(data_concat, lengths)
    
    '''
    lengths = [int(len(data_concat)*0.8), int(len(data_concat)*0.1),
               len(data_concat) - (int(len(data_concat)*0.8) + int(len(data_concat)*0.1))]
    train_data, val_data, test_data = torch.utils.data.random_split(data_concat, lengths)

    
    test_loader = DataLoader(test_data, batch_size=1, num_workers=1,
                            shuffle=False, collate_fn=lambda x: flickr_collate_style(x, 'romantic'))
    
    train_loader = DataLoader(train_data, batch_size=64, num_workers=24,
                            shuffle=False, collate_fn=lambda x: flickr_collate_style(x, 'romantic'))
    
    val_loader = DataLoader(val_data, batch_size=64, num_workers=24,
                            shuffle=False, collate_fn=lambda x: flickr_collate_style(x, 'romantic'))                     
    '''  
    train_loader = DataLoader(train_data, batch_size=64, num_workers=24,
                            shuffle=False, collate_fn= flickr_collate_fn)
                         
    val_loader = DataLoader(val_data, batch_size=64, num_workers=24,
                            shuffle=False, collate_fn=flickr_collate_fn)
    
    '''
    # model
    model = HyperNet(200, 200, 200, len(vocab), vocab, mixup=False)
    style_loss =True
    if style_loss:
        style_classifier = BertClassifer(vocab,  num_class=3).to(device)
        style_classifier.load_from_checkpoint(checkpoint_path=style_classifier_path, vocab=vocab).to(device)
    load_checkpoint = False
    if load_checkpoint:
        model.load_state_dict(torch.load(path_hn_humour))
        rnn = CaptionAttentionGru(200, 200, 200, len(vocab), vocab)
        rnn = rnn.load_from_checkpoint(checkpoint_path=gru_path, vocab=vocab)
        model.image_encoder = rnn.image_encoder
        model.captioner.feature_fc = rnn.captioner.feature_fc
        model.captioner.embed = rnn.captioner.embed
        model.captioner.fc = rnn.captioner.fc
        model.captioner.attention = rnn.captioner.attention
        model.captioner.init_h = rnn.captioner.init_h
    else:
        print('Loading GloVe Embedding')
        model.load_glove_emb(glove_path)
    print(model)

    wandb_logger = WandbLogger(save_dir='/cortex/users/cohenza4')
    lr_monitor_callback = pl.callbacks.LearningRateMonitor()
    checkpoint_callback = ModelCheckpoint(dirpath=save_path, monitor="val_loss with TF", save_top_k=1)
    print('Starting Training')
    trainer = pl.Trainer(gpus=[7], num_nodes=1, precision=32,
                         logger=wandb_logger,
                         #overfit_batches = 1,
                         check_val_every_n_epoch=1,
                         default_root_dir=save_path,
                         log_every_n_steps=20,
                         auto_lr_find=True,
                         callbacks=[lr_monitor_callback, checkpoint_callback],
                         #accelerator='ddp',
                         max_epochs=55,
                         gradient_clip_val=5.,
                         )
    trainer.tune(model, train_loader, val_loader)
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)
