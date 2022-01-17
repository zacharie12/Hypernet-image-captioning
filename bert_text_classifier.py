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
from models.decoderlstm import AttentionGru
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
from transformers import BertTokenizer
from transformers import BertModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BertClassifer(pl.LightningModule):
    def __init__(self, vocab, num_class=3, lr=1e-6, dropout=0.2):
        super(BertClassifer, self).__init__()

        self.vocab = vocab
        self.lr = lr
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.labels = {'factual':0,
                        'humour':1,
                        'romantic':2
                        }
         
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size*4),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.bert.config.hidden_size*4, num_class)
        )
     
        
    def load_glove_emb(self, emb_path):
        loader = WordVectorLoader(self.hparams['embed_size'])
        loader.load_glove(emb_path)
        emb_mat = loader.generate_embedding_matrix(self.vocab.w2i,
                                                   self.vocab.ix-1, 'norm')
        emb_mat = torch.FloatTensor(emb_mat)
        self.embedding = self.embedding.from_pretrained(emb_mat, freeze=True)

    def classes(self):
        return self.labels

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = F.softmax(linear_output)

        return final_layer

    def configure_optimizers(self):
        #params = list(self.linear.parameters())
        params = list(self.parameters())
        optimizer = torch.optim.Adam(params, lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, cooldown=2, factor=0.5)
        return [optimizer], [{'scheduler': scheduler, 'monitor': 'val_loss', 'interval': 'epoch'}]
        #return optimizer
  
    def training_step(self, train_batch, batch_idx):
        acc , total_count = 0, 0
        loss = torch.tensor(0.0).to(device)
        imgs, (style, (caps, lengths)) = train_batch
        label = torch.tensor([self.labels[style]]).to(device)
        for i in range(len(caps)):
            gt_idx = torch.squeeze(caps[i])
            text = cap_to_text_gt(gt_idx, self.vocab, tokenized=False)
            caption = self.tokenizer(text, 
                                padding='max_length', max_length = 25, truncation=True,
                                    return_tensors="pt") 

            mask = caption['attention_mask'].to(device)
            input_id = caption['input_ids'].squeeze(1).to(device)

            style_pred = self.forward(input_id, mask)
            loss +=  F.cross_entropy(style_pred, label)
            l2_lambda = 0.000001
            l2_reg = torch.tensor(0.).to(device)
            for param in self.parameters():
                l2_reg += torch.norm(param)
            loss += l2_lambda * l2_reg    
            acc += (style_pred.argmax(dim=1) == label).sum().item()

        loss /= len(caps)     
        mean_acc = acc / len(caps)
        self.log('train_loss', loss)
        self.log('Train accuracy', mean_acc)
        return loss

    def validation_step(self, val_batch, batch_idx):
        acc , total_count = 0, 0
        loss = torch.tensor(0.0).to(device)
        imgs, (style, (caps, lengths)) = val_batch
        label = torch.tensor([self.labels[style]]).to(device)
        for i in range(len(caps)):
            gt_idx = torch.squeeze(caps[i])
            text = cap_to_text_gt(gt_idx, self.vocab, tokenized=False)
            caption = self.tokenizer(text, 
                                padding='max_length', max_length = 25, truncation=True,
                                    return_tensors="pt") 

            mask = caption['attention_mask'].to(device)
            input_id = caption['input_ids'].squeeze(1).to(device)

            style_pred = self.forward(input_id, mask)
            loss +=  F.cross_entropy(style_pred, label)
            l2_lambda = 0.000001
            l2_reg = torch.tensor(0.).to(device)
            for param in self.parameters():
                l2_reg += torch.norm(param)
            loss += l2_lambda * l2_reg    
            acc += (style_pred.argmax(dim=1) == label).sum().item()

        loss /= len(caps)     
        mean_acc = acc / len(caps)
        self.log('val_loss', loss)
        self.log('val accuracy', mean_acc)

if __name__ == "__main__":
    img_path = "data/flickr7k_images"
    cap_path = "data/factual_train.txt"
    cap_path_humor = "data/humor/funny_train.txt"
    cap_path_romantic = "data/romantic/romantic_train.txt"
    glove_path = "/cortex/users/cohenza4/glove.6B.200d.txt"
    gru_path = "/cortex/users/cohenza4/checkpoint_gru/small_factual/epoch=43-step=1892.ckpt"
    save_path = "/cortex/users/cohenza4/checkpoint/style_classifier/"
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

    train_loader = DataLoader(train_data, batch_size=32, num_workers=12,
                              shuffle=False,  collate_fn=flickr_collate_fn)
    val_loader = DataLoader(val_data, batch_size=32, num_workers=12,
                            shuffle=False,  collate_fn=flickr_collate_fn)
    # model
    model = BertClassifer(vocab,  num_class=3, lr=1e-5)
    print('Loading GloVe Embedding')
    print(model)
    wandb_logger = WandbLogger(save_dir='/cortex/users/cohenza4')
    lr_monitor_callback = pl.callbacks.LearningRateMonitor()
    checkpoint_callback = ModelCheckpoint(dirpath=save_path, monitor="val_loss", save_top_k=1)
    print('Starting Training')
    trainer = pl.Trainer(gpus=[1], num_nodes=1, precision=32,
                         logger=wandb_logger,
                         check_val_every_n_epoch=1,
                         #overfit_batches=5,
                         default_root_dir=save_path,
                         log_every_n_steps=1,
                         auto_lr_find=False,
                         callbacks=[lr_monitor_callback, checkpoint_callback],
                         #max_epochs=50,
                         gradient_clip_val=5.,
                         #accelerator='ddp',
                         )                                
    trainer.tune(model, train_loader, val_loader)
    trainer.fit(model, train_loader, val_loader)
   