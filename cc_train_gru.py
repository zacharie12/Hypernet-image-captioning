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
from cc_dataloader import ConceptualCaptions, collate_fn, get_dataset
import build_vocab
from build_vocab import Vocab
import pickle
import random
from pytorch_lightning.loggers import WandbLogger   
from utils import set_all_parameters, flip_parameters_to_tensors, WordVectorLoader, cap_to_text, cap_to_text_gt, metric_score, metric_score_test, get_domain_list
from pytorch_lightning.callbacks import ModelCheckpoint
import datasets
import numpy as np
import random 
from hypernet_attention import HyperNet
import tldextract
from PIL import Image
import skimage.transform
import requests
import PIL
import cv2
from matplotlib import cm
from collections import Counter
from cider import Cider
from ptbtokenizer import PTBTokenizer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Gru(pl.LightningModule):
    def __init__(self, feature_size, embed_size, hidden_size, vocab_size, vocab, list_domain, lr=1e-6, domain_embed=False, zero_shot=False, list_zeroshot=[]):
        super().__init__()

        self.hparams['feature_size'] = feature_size
        self.hparams['vocab_size'] = vocab_size
        self.hparams['embed_size'] = embed_size
        self.hparams['hidden_size'] = hidden_size
        self.vocab = vocab
        self.hparams['lr'] = lr
        self.teacher_forcing_proba = 0.0
        self.beam_size = 3
        self.domain_embed = domain_embed
        self.Cider_scoreur = Cider()
        self.metrics = ['bleu', 'meteor', 'rouge']
        self.metrics = [datasets.load_metric(name) for name in self.metrics]

        self.dict_domain = {}
        self.tokenizer = PTBTokenizer()
        if domain_embed:
            for i in range(len(list_domain)):
                self.dict_domain[list_domain[i].replace("\n", '')] = i
                if zero_shot:
                    for j in range(len(list_zeroshot)):
                        self.dict_domain[list_zeroshot[j].replace("\n", '')] = j+100
            x = torch.tensor(list(self.dict_domain.values()))
            self.one_hot_emb = torch.nn.functional.one_hot(x , feature_size)
            self.captioner = AttentionGru(2048, feature_size, embed_size, hidden_size, vocab_size, 0, domain_embed, self.one_hot_emb,  self.dict_domain)
        else:
            self.captioner = AttentionGru(2048, feature_size, embed_size, hidden_size, vocab_size, 0 )
        self.image_encoder = EncoderCNN()


        
        

    def configure_optimizers(self):
        #params = list(self.captioner.parameters())
        params = list(self.captioner.gru.parameters())
        optimizer = torch.optim.Adam(params, lr=self.hparams['lr'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, cooldown=2, factor=0.5)
        return [optimizer], [{'scheduler': scheduler, 'monitor': 'val_loss with TF', 'interval': 'epoch'}]
        #return optimizer


    def forward(self):
        lstm_cop = self.captioner
        return lstm_cop


    def load_glove_emb(self, emb_path):
        loader = WordVectorLoader(self.hparams['embed_size'])
        loader.load_glove(emb_path)
        emb_mat = loader.generate_embedding_matrix(self.vocab.w2i, self.vocab.ix-1, 'norm')
        emb_mat = torch.FloatTensor(emb_mat)

        self.captioner.embed = self.captioner.embed.from_pretrained(emb_mat,
                                                                    freeze=True)

    def training_step(self, train_batch, batch_idx):
        imgs, caps, lengths, domains = train_batch
        domain = domains[0]
        img_feats = self.image_encoder(imgs.float()) 
        caps_pred, _ = self.captioner(img_feats, caps.long(), self.teacher_forcing_proba, domain)
        loss =  F.cross_entropy(caps_pred.view(-1, self.hparams['vocab_size']), caps.view(-1).long(), ignore_index=self.vocab.w2i['<pad>'])       
        bleu1, bleu2, bleu3, bleu4, meteor, rouge, cider = metric_score(caps, caps_pred, self.vocab, self.metrics)
  
        self.log('train_loss', loss)
        self.log('LR', self.hparams['lr'])
        self.log('TF', self.teacher_forcing_proba)
        self.log('meteor train', meteor)
        self.log('bleu 1 train', bleu1)
        self.log('bleu 2 train', bleu2)
        self.log('bleu 3 train', bleu3)
        self.log('bleu 4 train', bleu4)
        self.log('rouge train', rouge)
        self.log('cider train', cider)
        return loss


    def validation_step(self, val_batch, batch_idx):
        imgs, caps, lengths, domains = val_batch
        domain = domains[0]
        img_feats = self.image_encoder(imgs.float())
        caps_pred_teacher_forcing, _ = self.captioner(img_feats, caps.long(), self.teacher_forcing_proba, domain)
        caps_pred, _ = self.captioner(img_feats, caps.long(), 1.0, domain)

        loss = F.cross_entropy(caps_pred.view(-1, self.hparams['vocab_size']),
                               caps.view(-1).long(), ignore_index=self.vocab.w2i['<pad>'])
        
        bleu1, bleu2, bleu3, bleu4, meteor, rouge, cider = metric_score(caps, caps_pred, self.vocab, self.metrics)
         
        self.log('val_loss', loss, sync_dist=True)
        self.log('meteor val', meteor)
        self.log('bleu 1 val', bleu1)
        self.log('bleu 2 val', bleu2)
        self.log('bleu 3 val', bleu3)
        self.log('bleu 4 val', bleu4)
        self.log('rouge val', rouge)
        self.log('cider val', cider)
        loss_tf = F.cross_entropy(caps_pred_teacher_forcing.view(-1, self.hparams['vocab_size']),
                               caps.view(-1).long(), ignore_index=self.vocab.w2i['<pad>'])
        self.log('val_loss with TF', loss_tf, sync_dist=True)


    def test_step(self, test_batch, batch_idx):
        imgs, caps, lengths, domains = test_batch
        domain = domains[0]
        img_feats = self.image_encoder(imgs.float())
        caps_pred, _ = self.captioner(img_feats, caps.long(), 1.0, domain)

        bleu1, bleu2, bleu3, bleu4, meteor, rouge, cider = metric_score(caps, caps_pred, self.vocab, self.metrics)
         
        self.log('meteor test', meteor)
        self.log('bleu 1 test', bleu1)
        self.log('bleu 2 test', bleu2)
        self.log('bleu 3 test', bleu3)
        self.log('bleu 4 test', bleu4)
        self.log('rouge test', rouge)
        self.log('cider test', cider)

    '''
    def test_step(self, test_batch, batch_idx):
        imgs, caps, lengths, domains = test_batch
        domain = domains[0]
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
        if self.domain_embed:
            one_hot_emb = self.one_hot_emb[self.dict_domain[domain]]
            h = self.captioner.init_hidden(encoder_out, one_hot_emb)
        else:
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
        assert Caption_End
        if compute:
            indices = complete_seqs_scores.index(max(complete_seqs_scores))
            caps_pred_beam = complete_seqs[indices]




            bleu1_beam, bleu2_beam, bleu3_beam, bleu4_beam, meteor_beam, rouge_beam, cider_beam = metric_score_test(caps, torch.tensor(caps_pred_beam), self.vocab, self.metrics)
            self.log('meteor beam', meteor_beam)
            self.log('bleu 1 beam', bleu1_beam)
            self.log('bleu 2 beam', bleu2_beam)
            self.log('bleu 3 beam', bleu3_beam)
            self.log('bleu 4 beam', bleu4_beam)
            self.log('rouge beam', rouge_beam)
            self.log('cider beam', cider_beam)

        img_feats = self.image_encoder(imgs.float())
        caps_pred, _ = self.captioner(img_feats, caps.long(), 1.0, domain)
        bleu1, bleu2, bleu3, bleu4, meteor, rouge, cider = metric_score(caps, caps_pred, self.vocab, self.metrics)
        self.log('meteor test', meteor)
        self.log('bleu 1 test', bleu1)
        self.log('bleu 2 test', bleu2)
        self.log('bleu 3 test', bleu3)
        self.log('bleu 4 test', bleu4)
        self.log('rouge test', rouge)
        self.log('cider test', cider)


    '''
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('Gru')
        '''
        parser.add_argument('--img_dir_train', type=str, default='data/200_conceptual_images_train/')
        parser.add_argument('--img_dir_val_test', type=str, default="data/200_conceptual_images_val/")
        parser.add_argument('--cap_dir_train', type=str, default='data/train_cap_100.txt')
        parser.add_argument('--cap_dir_val', type=str, default="data/val_cap_100.txt")
        parser.add_argument('--cap_dir_test', type=str, default="data/test_cap_100.txt")
        parser.add_argument('--glove_path', type=str, default="/cortex/users/cohenza4/glove.6B.200d.txt")
        parser.add_argument('--vocab_path', type=str, default="data/vocab.pkl")
        parser.add_argument('--save_path', type=str, default='/cortex/users/cohenza4/checkpoint/GRU/')
        parser.add_argument('--batch_size', type=int, default=32)
        '''
        parser.add_argument('--domain_emb', type=bool, default=False)
        parser.add_argument('--max_epochs', type=int, default=50)

if __name__ == "__main__":
    #out_cap = train_cap_100.txt
    #out_img = 200_conceptual_images_train
    glove_path = "/cortex/users/cohenza4/glove.6B.200d.txt"
    img_dir_train = 'data/cc_big_100/'
    img_dir_val_test = 'data/200_conceptual_images_val/'
    cap_dir_train = 'data/cc_big_100.txt'
    cap_dir_val = 'data/val_cap_100.txt'
    cap_dir_test = 'data/test_cap_100.txt'
    save_path = "/cortex/users/cohenza4/checkpoint_CC/GRU_big/emb/"
    # data
    
    with open("data/vocab.pkl", 'rb') as f:
        vocab = pickle.load(f)

    train_data = get_dataset(img_dir_train, cap_dir_train, vocab)
    val_data = get_dataset(img_dir_val_test, cap_dir_val, vocab)
    test_data = get_dataset(img_dir_val_test, cap_dir_test, vocab)
    train_loader = DataLoader(train_data, batch_size=32, num_workers=2,
                            shuffle=False, collate_fn= collate_fn)
    val_loader = DataLoader(val_data, batch_size=10, num_workers=2,
                            shuffle=False, collate_fn= collate_fn)
    test_loader = DataLoader(test_data, batch_size=20, num_workers=2,
                            shuffle=False, collate_fn= collate_fn)
    list_domain = get_domain_list(cap_dir_train, cap_dir_val)
    domain_emb = True
    model = Gru(200, 200, 200, len(vocab), vocab, list_domain, 0.001, domain_emb)                       
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
                         auto_lr_find=False,
                         callbacks=[lr_monitor_callback, checkpoint_callback],
                         #accelerator='ddp',
                         max_epochs=30,
                         gradient_clip_val=5.,
                         )
    trainer.tune(model, train_loader, val_loader)
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)

    

    