
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
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import copy
from data_loader import get_dataset, get_styled_dataset, flickr_collate_fn, ConcatDataset, FlickrCombiner, flickr_collate_style
import build_vocab
from build_vocab import Vocab
import pickle
import random
from pytorch_lightning.loggers import WandbLogger   
from utils import set_all_parameters, flip_parameters_to_tensors, WordVectorLoader, cap_to_text, cap_to_text_gt, metric_score
from pytorch_lightning.callbacks import ModelCheckpoint
import datasets
import numpy as np

class HyperNet(pl.LightningModule):
    def __init__(self, feature_size, embed_size, hidden_size, vocab_size, vocab, num_layers=1, lr=1e-6):
        super().__init__()

        self.hparams['feature_size'] = feature_size
        self.hparams['vocab_size'] = vocab_size
        self.hparams['embed_size'] = embed_size
        self.hparams['hidden_size'] = hidden_size
        self.vocab = vocab
        self.hparams['lr'] = lr
        self.hparams['num_layers'] = num_layers
        self.teacher_forcing_proba = 0.0
        self.metrics = ['bleu', 'meteor', 'rouge']
        self.metrics = [datasets.load_metric(name) for name in self.metrics]

        # create image encoder
        self.image_encoder = EncoderCNN()
            
        self.captioner = AttentionGru(2048, feature_size, embed_size, hidden_size, vocab_size, num_layers=num_layers, p=0.0)
        # create HN base
        N = 2
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
                    nn.Linear(N*embed_size, N_size),
                    nn.LeakyReLU(),
                    nn.Linear(w_size, w_size)
                ))
            else:
                if w_size // 16 < N*embed_size:
                    self.hn_heads.append(nn.Sequential(
                        nn.Linear(N*embed_size, N*embed_size),
                        nn.LeakyReLU(),
                        nn.Linear(N*embed_size, w_size)
                    ))
                else:
                    self.hn_heads.append(nn.Sequential(
                        nn.Linear(N*embed_size, w_size//16),
                        nn.LeakyReLU(),
                        nn.Linear(w_size//16, w_size)
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
        style = torch.tensor([vocab(style)])
        style = style.type(torch.LongTensor)
        style = style.to(self.device)
        style_embed = self.captioner.embed(style)

        self.forward(style_embed)
        img_feats = self.image_encoder(imgs.float())
        caps_pred, _ = self.captioner(img_feats, caps.long(), self.teacher_forcing_proba)
        loss =  F.cross_entropy(caps_pred.view(-1, self.hparams['vocab_size']), caps.view(-1).long(), ignore_index=self.vocab.w2i['<pad>'])        
        bleu, meteor, rouge = metric_score(caps, caps_pred, self.vocab, self.metrics)
        self.log('train_loss', loss)
        self.log('LR', self.hparams['lr'])
        if self.teacher_forcing_proba < 0.75: 
            self.teacher_forcing_proba = self.teacher_forcing_proba * 1.00005
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
        
        bleu, meteor, rouge = metric_score(caps, caps_pred, self.vocab, self.metrics)
        
        self.log('val_loss', loss, sync_dist=True)
        self.log('meteor', meteor)
        self.log('bleu', bleu)

        loss_tf = F.cross_entropy(caps_pred_teacher_forcing.view(-1, self.hparams['vocab_size']),
                               caps.view(-1).long(), ignore_index=self.vocab.w2i['<pad>'])
        self.log('val_loss with TF', loss_tf, sync_dist=True)
        

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
    glove_path = "/cortex/users/cohenza4/glove.6B.100d.txt"
    path_factual= "/cortex/users/cohenza4/checkpoint_gru/hn/hn_model.ckpt"
    path_all = "/cortex/users/cohenza4/checkpoint_gru/hn/hn_all_tf.pt"
    gru_path = "/cortex/users/cohenza4/checkpoint_gru/small_factual/epoch=43-step=1892.ckpt"
    save_path = "/cortex/users/cohenza4/checkpoint_gru/HN/factual1/"
    # data
    with open("data/vocab.pkl", 'rb') as f:
        vocab = pickle.load(f)
    orig_dataset = get_dataset(img_path, cap_path, vocab)
    humor_dataset = get_styled_dataset(cap_path_humor, vocab)
    romantic_dataset = get_styled_dataset(cap_path_romantic, vocab)

    data_concat = ConcatDataset(orig_dataset, humor_dataset, romantic_dataset)
    lengths = [int(len(data_concat)*0.8),
            len(data_concat) - int(len(data_concat)*0.8)]
    train_data, val_data = torch.utils.data.random_split(data_concat, lengths)
    '''
    
    train_loader = DataLoader(train_data, batch_size=128, num_workers=24,
                            shuffle=False, collate_fn=lambda x: flickr_collate_style(x, 'factual'))
    '''
    val_loader = DataLoader(val_data, batch_size=128, num_workers=24,
                            shuffle=False, collate_fn=lambda x: flickr_collate_style(x, 'romantic'))                     
    
    train_loader = DataLoader(train_data, batch_size=128, num_workers=24,
                            shuffle=False, collate_fn= flickr_collate_fn)
    '''                      
    val_loader = DataLoader(val_data, batch_size=128, num_workers=24,
                            shuffle=False, collate_fn=flickr_collate_fn)
    
    '''  
    # model
    model = HyperNet(200, 100, 180, len(vocab), vocab)
    load_checkpoint = True
    if load_checkpoint:
        model.load_state_dict(torch.load(path_all))
        rnn = CaptionAttentionGru(200, 100, 180, len(vocab), vocab)
        rnn = rnn.load_from_checkpoint(checkpoint_path=gru_path, vocab=vocab)
        model.captioner.embed = rnn.captioner.embed
        model.captioner.fc = rnn.captioner.fc
        model.captioner.attention = rnn.captioner.attention
        model.captioner.init_h = rnn.captioner.init_h
    else:
        print('Loading GloVe Embedding')
        #model.load_glove_emb(glove_path)
    #print(model)

    wandb_logger = WandbLogger(save_dir='/cortex/users/cohenza4')
    lr_monitor_callback = pl.callbacks.LearningRateMonitor()
    checkpoint_callback = ModelCheckpoint(dirpath=save_path, monitor="val_loss", save_top_k=1)
    print('Starting Training')
    trainer = pl.Trainer(gpus=[5], num_nodes=1, precision=32,
                         logger=wandb_logger,
                         #overfit_batches = 1,
                         check_val_every_n_epoch=1,
                         default_root_dir=save_path,
                         log_every_n_steps=20,
                         auto_lr_find=True,
                         callbacks=[lr_monitor_callback, checkpoint_callback],
                         #accelerator='ddp',
                         gradient_clip_val=5.,
                         )
    trainer.tune(model, train_loader, val_loader)
    trainer.fit(model, train_loader, val_loader)
