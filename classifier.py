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
from models.decoderlstm import AttentionGru, BeamSearch
from models.encoder import EncoderCNN
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import copy
import numpy as np
from data_loader import get_dataset, get_styled_dataset, flickr_collate_fn, ConcatDataset, FlickrCombiner, flickr_collate_style, collate_fn_classifier
import build_vocab
from build_vocab import Vocab
import pickle
import random
from pytorch_lightning.loggers import WandbLogger   
from pytorch_lightning.callbacks import ModelCheckpoint
from utils import set_all_parameters, flip_parameters_to_tensors, WordVectorLoader, cap_to_text, cap_to_text_gt, metric_score
import datasets
import dominate
from dominate.tags import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class StyleClassifer(pl.LightningModule):
    def __init__(self, vocab_size,vocab, embed_dim, num_class=3, lr=1e-6):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.vocab = vocab
        self.lr = lr
         
        self.embedding = nn.Sequential(
            nn.Embedding(vocab_size, embed_dim),
            nn.LeakyReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LeakyReLU()
        )
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()
            
        
    def load_glove_emb(self, emb_path):
        loader = WordVectorLoader(self.hparams['embed_size'])
        loader.load_glove(emb_path)
        emb_mat = loader.generate_embedding_matrix(self.vocab.w2i,
                                                   self.vocab.ix-1, 'norm')
        emb_mat = torch.FloatTensor(emb_mat)
        self.embedding = self.embedding.from_pretrained(emb_mat, freeze=True)

    def init_weights(self):
        initrange = 0.5
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text):
        embedded = self.embedding(text)
        return F.softmax(self.fc(embedded))

    def configure_optimizers(self):
        params = list(self.parameters())
        optimizer = torch.optim.Adam(params, lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, cooldown=2, factor=0.5)
        return [optimizer], [{'scheduler': scheduler, 'monitor': 'val_loss', 'interval': 'epoch'}]
        #return optimizer
  
    def training_step(self, train_batch, batch_idx):
        total_acc, total_count = 0, 0

        imgs, (style, (caps, lengths)) = train_batch
        if style == "factual":
            gt_style = torch.tensor([[1.0, 0.0,0.0]]).repeat(caps.size()[0],1).to(device)  
        elif style == "humour":
            gt_style = torch.tensor([[0.0, 1.0,0.0]]).repeat(caps.size()[0],1).to(device)  
        else:
            gt_style = torch.tensor([[0.0, 0.0,1.0]]).repeat(caps.size()[0],1).to(device)            

        style_pred = self.forward(caps.long())
        loss =  F.cross_entropy(style_pred, gt_style.long())    
        pred = style_pred.argmax(1)    
        total_acc += (pred == gt_style).sum().item()
        total_count += gt_style.size(0)
        acc = total_acc/total_count
        self.log('train loss', loss)
        self.log('Train accuracy', acc)
        total_acc, total_count = 0, 0
        return loss

    def validation_step(self, val_batch, batch_idx):
        total_acc, total_count = 0, 0
        imgs, (style, (caps, lengths)) = val_batch
        if style == "factual":
            gt_style = torch.tensor([[1.0, 0.0,0.0]]).repeat(caps.size()[0],1).to(device)  
        elif style == "humour":
            gt_style = torch.tensor([[0.0, 1.0,0.0]]).repeat(caps.size()[0],1).to(device)  
        else:
            gt_style = torch.tensor([[0.0, 0.0,1.0]]).repeat(caps.size()[0],1).to(device)            


        style_pred = self.forward(caps.long())
        loss =  F.cross_entropy(style_pred, gt_style.long())        
        total_acc += (style_pred.argmax(1) == gt_style).sum().item()
        total_count += gt_style.size(0)
        acc = total_acc/total_count
        self.log('val_loss', loss)
        self.log('validation accuracy', acc)
        total_acc, total_count = 0, 0

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
    glove_path = "/cortex/users/cohenza4/glove.6B.200d.txt"
    gru_path = "/cortex/users/cohenza4/checkpoint_gru/small_factual/epoch=43-step=1892.ckpt"
    save_path = "/cortex/users/cohenza4/checkpoint_gru/classifier/"
    # data
    with open("data/vocab.pkl", 'rb') as f:
        vocab = pickle.load(f)
    print('Prepairing Data')
    orig_dataset = get_dataset(img_path, cap_path, vocab)
    humor_dataset = get_styled_dataset(cap_path_humor, vocab)
    romantic_dataset = get_styled_dataset(cap_path_romantic, vocab)

    data_concat = ConcatDataset(orig_dataset, humor_dataset, romantic_dataset)
 
    lengths = [int(len(data_concat)*0.8),
               len(data_concat) - int(len(data_concat)*0.8)]
    train_data, val_data = torch.utils.data.random_split(data_concat, lengths)

    train_loader = DataLoader(train_data, batch_size=3, num_workers=12,
                              shuffle=False,  collate_fn=flickr_collate_fn)
    val_loader = DataLoader(val_data, batch_size=3, num_workers=12,
                            shuffle=False,  collate_fn=flickr_collate_fn)
    # model
    model = StyleClassifer(len(vocab), vocab, embed_dim=200, num_class=3, lr=0.0003)
    print('Loading GloVe Embedding')
    #model.load_glove_emb(glove_path)
    print(model)
    wandb_logger = WandbLogger(save_dir='/cortex/users/cohenza4')
    lr_monitor_callback = pl.callbacks.LearningRateMonitor()
    checkpoint_callback = ModelCheckpoint(dirpath=save_path, monitor="val_loss", save_top_k=1)
    print('Starting Training')
    trainer = pl.Trainer(gpus=[6], num_nodes=1, precision=32,
                         logger=wandb_logger,
                         check_val_every_n_epoch=1,
                         #overfit_batches=5,
                         default_root_dir=save_path,
                         log_every_n_steps=20,
                         auto_lr_find=True,
                         callbacks=[lr_monitor_callback, checkpoint_callback],
                         #max_epochs=50,
                         gradient_clip_val=5.,
                         #accelerator='ddp',
                         )                                
    trainer.tune(model, train_loader, val_loader)
    trainer.fit(model, train_loader, val_loader)
   