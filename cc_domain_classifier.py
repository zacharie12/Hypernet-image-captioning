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
from cc_dataloader import ConceptualCaptions, collate_fn, get_dataset
import build_vocab
from build_vocab import Vocab
import pickle
import random
from pytorch_lightning.loggers import WandbLogger   
from pytorch_lightning.callbacks import ModelCheckpoint
from utils import set_all_parameters, flip_parameters_to_tensors, WordVectorLoader, cap_to_text, cap_to_text_gt, metric_score, metric_score_test, get_domain_list, get_hist_embedding, tfidf_hist, get_jsd_tsne
from pytorch_lightning.callbacks import ModelCheckpoint
import datasets
import dominate
from dominate.tags import *
from transformers import BertTokenizer
from transformers import BertModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BertClassiferCC(pl.LightningModule):
    def __init__(self, vocab, domain_list, lr=1e-6, dropout=0.2):
        super(BertClassiferCC, self).__init__()

        self.vocab = vocab
        self.lr = lr
        self.domain_list = domain_list
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.labels = {}
        self.num_class = len(domain_list)
        for i in range(len(domain_list)):
                self.labels[domain_list[i].replace("\n", '')] = i
         
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size*4),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.bert.config.hidden_size*4, self.num_class)
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
        imgs, caps, lengths, domains = train_batch
        domain = domains[0]
        label = torch.tensor([self.labels[domain]]).to(device)
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
            l2_lambda = 0.0000001
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
        imgs, caps, lengths, domains = val_batch
        domain = domains[0]
        label = torch.tensor([self.labels[domain]]).to(device)
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
    glove_path = "/cortex/users/cohenza4/glove.6B.200d.txt"
    img_dir_train = 'data/200_conceptual_images_train/'
    img_dir_val_test = 'data/200_conceptual_images_val/'
    cap_dir_train = 'data/train_cap_100.txt'
    cap_dir_val = 'data/val_cap_100.txt'
    cap_dir_test = 'data/test_cap_100.txt'
    save_path = "/cortex/users/cohenza4/checkpoint/domain_classifier/"
    # data
    with open("data/vocab.pkl", 'rb') as f:
        vocab = pickle.load(f)
    train_data = get_dataset(img_dir_train, cap_dir_train, vocab)
    val_data = get_dataset(img_dir_val_test, cap_dir_val, vocab)

    train_loader = DataLoader(train_data, batch_size=32, num_workers=2,
                            shuffle=False, collate_fn= collate_fn)
    val_loader = DataLoader(val_data, batch_size=8, num_workers=2,
                            shuffle=False, collate_fn= collate_fn)
    domain_list = get_domain_list(cap_dir_train, cap_dir_val)
    # model
    model = BertClassiferCC(vocab, domain_list, lr=1e-6)
    print('Loading GloVe Embedding')
    print(model)
    wandb_logger = WandbLogger(save_dir='/cortex/users/cohenza4')
    lr_monitor_callback = pl.callbacks.LearningRateMonitor()
    checkpoint_callback = ModelCheckpoint(dirpath=save_path, monitor="val_loss", save_top_k=1)
    print('Starting Training')
    trainer = pl.Trainer(gpus=[7], num_nodes=1, precision=32,
                         logger=wandb_logger,
                         check_val_every_n_epoch=1,
                         overfit_batches=5,
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
   