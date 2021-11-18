from numpy.core.numeric import False_
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.models as models
from torch.utils import data
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torchvision
from torchvision import transforms
from models.decoderlstm import AttentionLstm
from models.encoder import EncoderCNN
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import copy
import numpy as np
from data_loader import get_dataset, get_styled_dataset, flickr_collate_fn, ConcatDataset, FlickrCombiner, flickr_collate_style
import build_vocab
from build_vocab import Vocab
import pickle
import random
from pytorch_lightning.loggers import WandbLogger   
from pytorch_lightning.callbacks import ModelCheckpoint
from utils import set_all_parameters, flip_parameters_to_tensors, WordVectorLoader, cap_to_text, cap_to_text_gt, metric_score
import datasets


class CaptionAttentionLstm(pl.LightningModule):
    def __init__(self, features_size, embed_size, hidden_size, vocab_size, vocab, lr=1e-6, sample_prob=0.5):
        super().__init__()

        self.hparams['vocab_size'] = vocab_size
        self.hparams['embed_size'] = embed_size
        self.hparams['hidden_size'] = hidden_size
        self.hparams['features_size'] = features_size
        self.vocab = vocab
        self.hparams['lr'] = lr
        self.sample_prob = sample_prob
        self.teacher_forcing_proba = 0.0
    
        # create image encoder 
        self.image_encoder = EncoderCNN()           
        self.captioner = AttentionLstm(features_size, embed_size, hidden_size, vocab_size, p=0.5)
        self.metrics = ['bleu', 'meteor', 'rouge']
        self.metrics = [datasets.load_metric(name) for name in self.metrics]


    def load_glove_emb(self, emb_path):
        loader = WordVectorLoader(self.hparams['embed_size'])
        loader.load_glove(emb_path)
        emb_mat = loader.generate_embedding_matrix(self.vocab.w2i,
                                                   self.vocab.ix-1, 'norm')
        emb_mat = torch.FloatTensor(emb_mat)
        self.captioner.embeddings = self.captioner.embeddings.from_pretrained(emb_mat,
                                                                    freeze=True)
    
    def forward(self, img, caps, sample_prob=1.0):
        features = self.image_encoder(img.float())
        caps_pred, _ = self.captioner(caps.long(), features, sample_prob)
        return caps_pred

    def configure_optimizers(self):
        params = list(self.captioner.parameters())
        #params = list(self.captioner.gru.parameters())
        optimizer = torch.optim.Adam(params, lr=self.hparams['lr'])
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, cooldown=2)
        #return [optimizer], [{'scheduler': scheduler, 'monitor': 'val_loss'}]
        return optimizer
  
    def training_step(self, train_batch, batch_idx):
        imgs, (style, (caps, lengths)) = train_batch
        caps_pred = self.forward(imgs, caps, self.teacher_forcing_proba)
        loss =  F.cross_entropy(caps_pred.view(-1, self.hparams['vocab_size']), caps.view(-1).long(), ignore_index=self.vocab.w2i['<pad>'])        
        bleu, meteor, rouge = metric_score(caps, caps_pred, self.vocab, self.metrics)
        self.log('train_loss', loss)
        self.log('LR', self.hparams['lr'])
        self.log('TF', self.teacher_forcing_proba)
        if self.teacher_forcing_proba < 0.75: 
            self.teacher_forcing_proba = self.teacher_forcing_proba * 1.001
        return loss

    def validation_step(self, val_batch, batch_idx):
        imgs, (style, (caps, lengths)) = val_batch
        caps_pred_teacher_forcing = self.forward(imgs, caps, self.teacher_forcing_proba)
        caps_pred = self.forward(imgs, caps, 1.0)
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
        parser.add_argument("--lr", type=float, default=5e-3)
        return parent_parser

        
if __name__ == "__main__":
    img_path = "data/flickr7k_images"
    cap_path = "data/factual_train.txt"
    cap_path_humor = "data/humor/funny_train.txt"
    cap_path_romantic = "data/romantic/romantic_train.txt"
    glove_path = "/cortex/users/cohenza4/glove.6B.300d.txt"
    gru_path = "/cortex/users/cohenza4/checkpoint_gru/factual/epoch=6-step=819.ckpt"
    save_path = "/cortex/users/cohenza4/checkpoint_lstm/factual/"
    # data
    with open("data/vocab.pkl", 'rb') as f:
        vocab = pickle.load(f)
    print('Prepairing Data')
    orig_dataset = get_dataset(img_path, cap_path, vocab)
    humor_dataset = get_styled_dataset(cap_path_humor, vocab)
    romantic_dataset = get_styled_dataset(cap_path_romantic, vocab)

    data_concat = ConcatDataset(orig_dataset, humor_dataset, romantic_dataset)
    lengths = [int(len(data_concat)*0.9),
               len(data_concat) - int(len(data_concat)*0.9)]
    train_data, val_data = torch.utils.data.random_split(data_concat, lengths)

    train_loader = DataLoader(train_data, batch_size=64, num_workers=32,
                              shuffle=False, collate_fn=lambda x: flickr_collate_style(x, 'humor'))
    val_loader = DataLoader(val_data, batch_size=1, num_workers=1,
                            shuffle=False, collate_fn=lambda x: flickr_collate_style(x, 'humor'))
    # model
    model = CaptionAttentionLstm(2048, 300, 512, len(vocab), vocab, lr=0.0002)
    load_checkpoint = False
    if load_checkpoint:
        rnn = CaptionAttentionLstm(100, 300, 150, len(vocab), vocab, 1, lr=5e-6)
        rnn = rnn.load_from_checkpoint(checkpoint_path=gru_path, vocab=vocab)
        model.captioner.embed = rnn.captioner.embed
        model.captioner.fc = rnn.captioner.fc
        model.captioner.attention = rnn.captioner.attention
        model.captioner.init_h = rnn.captioner.init_h
    else:
        print('Loading GloVe Embedding')
        model.load_glove_emb(glove_path)
    print(model)
    #model.load_glove_emb(glove_path)


    wandb_logger = WandbLogger(save_dir='/cortex/users/cohenza4')
    #wandb_logger.log_hyperparams(model.hparams)
    lr_monitor_callback = pl.callbacks.LearningRateMonitor()
    checkpoint_callback = ModelCheckpoint(dirpath=save_path, monitor="val_loss")
    print('Starting Training')
    trainer = pl.Trainer(gpus=[0], num_nodes=1, precision=32,
                         logger=wandb_logger,
                         check_val_every_n_epoch=1,
                         #overfit_batches=1,
                         default_root_dir=save_path,
                         log_every_n_steps=20,
                         auto_lr_find=False,
                         callbacks=[lr_monitor_callback, checkpoint_callback],
                         #callbacks=[lr_monitor_callback],
                         gradient_clip_val=5.,
                         #accelerator='ddp',
                         )                                
    trainer.tune(model, train_loader, val_loader)
    trainer.fit(model, train_loader, val_loader)
