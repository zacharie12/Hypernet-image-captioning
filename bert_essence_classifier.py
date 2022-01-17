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
from models.decoderlstm import AttentionGru, BeamSearch, classifier_end
from models.encoder import EncoderCNN
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import copy
import numpy as np
from data_loader import get_dataset, get_styled_dataset, flickr_collate_fn, ConcatDataset, FlickrCombiner, flickr_collate_style, collate_fn_classifier, flickr_collate_fn_essence
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

class BertEssenceClassifer(pl.LightningModule):
    def __init__(self, vocab, feature_size=200, num_class=2, lr=1e-6, dropout=0.2):
        super(BertEssenceClassifer, self).__init__()

        self.vocab = vocab
        self.lr = lr
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.feature_size = feature_size
        self.linear = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size*4),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.bert.config.hidden_size*4, self.feature_size)
        )
        self.classifier_end  = classifier_end(feature_size)
        self.labels = {'same img':1,
                       'diff img':0
                        }
        
    def load_glove_emb(self, emb_path):
        loader = WordVectorLoader(self.hparams['embed_size'])
        loader.load_glove(emb_path)
        emb_mat = loader.generate_embedding_matrix(self.vocab.w2i,
                                                   self.vocab.ix-1, 'norm')
        emb_mat = torch.FloatTensor(emb_mat)
        self.embedding = self.embedding.from_pretrained(emb_mat, freeze=True)

    def classes(self):
        return self.labels

    def forward(self, input_id1, mask1, input_id2, mask2):

        _, pooled_output1 = self.bert(input_ids= input_id1, attention_mask=mask1,return_dict=False)
        dropout_output1 = self.dropout(pooled_output1)
        linear_output1 = self.linear(dropout_output1)
        _, pooled_output2 = self.bert(input_ids= input_id2, attention_mask=mask2,return_dict=False)
        dropout_output2 = self.dropout(pooled_output2)
        linear_output2 = self.linear(dropout_output2)
        final_layer = self.classifier_end(linear_output1, linear_output2)
        return final_layer

    def configure_optimizers(self):
        #params = list(self.linear.parameters())
        params = list(self.parameters())
        optimizer = torch.optim.Adam(params, lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, cooldown=2, factor=0.5)
        return [optimizer], [{'scheduler': scheduler, 'monitor': 'val_loss', 'interval': 'epoch'}]
        #return optimizer
  
    def training_step(self, train_batch, batch_idx):
        total_acc_train , total_count, cnt_label_pos, acc_pos, acc_neg = 0, 0, 0, 0, 0
        loss = torch.tensor(0.0).to(device)
        mask_fac, mask_hum, mask_rom, input_id_fac, input_id_hum, input_id_rom = [], [], [], [], [], []
        imgs,((style_fac,(cap_fac,len_fac)), (style_hum,(cap_hum,len_hum)),(style_rom,(cap_rom,len_rom))) = train_batch
        for i in range(len(cap_fac)):
            gt_idx_fac = torch.squeeze(cap_fac[i])
            gt_idx_hum = torch.squeeze(cap_hum[i])
            gt_idx_rom = torch.squeeze(cap_rom[i])
            text_fact = cap_to_text_gt(gt_idx_fac, self.vocab, tokenized=False)
            text_hum = cap_to_text_gt(gt_idx_hum, self.vocab, tokenized=False)
            text_rom = cap_to_text_gt(gt_idx_rom, self.vocab, tokenized=False)

            caption_fac = self.tokenizer(text_fact, 
                                padding='max_length', max_length = 25, truncation=True,
                                    return_tensors="pt") 
            caption_hum = self.tokenizer(text_hum, 
                                padding='max_length', max_length = 25, truncation=True,
                                    return_tensors="pt") 
            caption_rom = self.tokenizer(text_rom, 
                                padding='max_length', max_length = 25, truncation=True,
                                    return_tensors="pt")

            mask_fac.append(caption_fac['attention_mask'].to(device))
            mask_hum.append(caption_hum['attention_mask'].to(device))
            mask_rom.append(caption_rom['attention_mask'].to(device))
            input_id_fac.append(caption_fac['input_ids'].squeeze(1).to(device))
            input_id_hum.append(caption_hum['input_ids'].squeeze(1).to(device))
            input_id_rom.append(caption_rom['input_ids'].squeeze(1).to(device))


        #loop  all the combination with 3 style and and n images when it's the same image label is 1 else 0
        n = len(cap_fac)
        #num_com = (3*n) * (3*n -1)/2
        for i in range(n):
            j = (i + 1) % n
            gt = 'same img'
            label = torch.tensor([self.labels[gt]]).to(device)
            input_id1 = input_id_fac[i]
            mask1 = mask_fac[i]
            input_id2 = input_id_hum[i]
            mask2 = mask_hum[i]
            pred = self.forward(input_id1, mask1, input_id2, mask2)
            pred_label_int = 1 if pred > 0.0 else 0
            pred_label = torch.tensor([pred_label_int]).to(device)
            if pred_label == label:
                total_acc_train += 1 
            total_count += 1
            pred = pred.to(torch.float32)
            label = label.to(torch.float32)
            loss +=  F.mse_loss(pred, label) 

            gt = 'diff img'
            label = torch.tensor([self.labels[gt]]).to(device)
            style_2 = random.randint(0,2)
            if style_2 == 0:
                input_id2 = input_id_fac[j]
                mask2 = mask_fac[j]
            elif style_2 == 1:
                input_id2 = input_id_hum[j] 
                mask2 = mask_hum[j] 
            else:
                input_id2 = input_id_rom[j] 
                mask2 = mask_rom[j]
            pred_label_int = 1 if pred > 0.0 else 0
            pred_label = torch.tensor([pred_label_int]).to(device)
            if pred_label == label:
                total_acc_train += 1 
            total_count += 1
            label = label.to(torch.float32)
            loss +=  F.mse_loss(pred, label) 

            gt = 'same img'
            label = torch.tensor([self.labels[gt]]).to(device)
            input_id2 = input_id_rom[i]
            mask2 = mask_rom[i]
            pred = self.forward(input_id1, mask1, input_id2, mask2)
            pred_label_int = 1 if pred > 0.0 else 0
            pred_label = torch.tensor([pred_label_int]).to(device)
            if pred_label == label:
                total_acc_train += 1 
            total_count += 1
            label = label.to(torch.float32)
            loss +=  F.mse_loss(pred, label) 

            gt = 'diff img'
            label = torch.tensor([self.labels[gt]]).to(device)
            style_2 = random.randint(0,2)
            if style_2 == 0:
                input_id1 = input_id_fac[j]
                mask1 = mask_fac[j]
            elif style_2 == 1:
                input_id1 = input_id_hum[j] 
                mask1 = mask_hum[j] 
            else:
                input_id1 = input_id_rom[j] 
                mask1 = mask_rom[j]
            pred_label_int = 1 if pred > 0.0 else 0
            pred_label = torch.tensor([pred_label_int]).to(device)
            if pred_label == label:
                total_acc_train += 1 
            total_count += 1
            label = label.to(torch.float32)
            loss +=  F.mse_loss(pred, label) 

            gt = 'same img'
            label = torch.tensor([self.labels[gt]]).to(device)
            input_id1 = input_id_hum[i]
            mask1 = mask_hum[i]
            pred = self.forward(input_id1, mask1, input_id2, mask2)
            pred_label_int = 1 if pred > 0.0 else 0
            pred_label = torch.tensor([pred_label_int]).to(device)
            if pred_label == label:
                total_acc_train += 1 
            total_count += 1
            label = label.to(torch.float32)
            loss +=  F.mse_loss(pred, label) 

            gt = 'diff img'
            label = torch.tensor([self.labels[gt]]).to(device)
            style_2 = random.randint(0,2)
            if style_2 == 0:
                input_id2 = input_id_fac[j]
                mask2 = mask_fac[j]
            elif style_2 == 1:
                input_id2 = input_id_hum[j] 
                mask2 = mask_hum[j] 
            else:
                input_id2 = input_id_rom[j] 
                mask2 = mask_rom[j]
            pred_label_int = 1 if pred > 0.0 else 0
            pred_label = torch.tensor([pred_label_int]).to(device)
            if pred_label == label:
                total_acc_train += 1 
            total_count += 1
            label = label.to(torch.float32)
            loss +=  F.mse_loss(pred, label) 

            l2_lambda = 0.000001
            l2_reg = torch.tensor(0.).to(device)
            for param in self.parameters():
                l2_reg += torch.norm(param)
            loss += l2_lambda * l2_reg 
            
        total_acc_train /= total_count
        self.log('train_loss', loss)
        self.log('Train accuracy', total_acc_train)
        return loss

    def validation_step(self, val_batch, batch_idx):
        total_acc , total_count, cnt_label_pos = 0, 0, 0
        loss = torch.tensor(0.0).to(device)
        mask_fac, mask_hum, mask_rom, input_id_fac, input_id_hum, input_id_rom = [], [], [], [], [], []
        imgs,((style_fac,(cap_fac,len_fac)), (style_hum,(cap_hum,len_hum)),(style_rom,(cap_rom,len_rom))) = val_batch
        for i in range(len(cap_fac)):
            gt_idx_fac = torch.squeeze(cap_fac[i])
            gt_idx_hum = torch.squeeze(cap_hum[i])
            gt_idx_rom = torch.squeeze(cap_rom[i])
            text_fact = cap_to_text_gt(gt_idx_fac, vocab, tokenized=False)
            text_hum = cap_to_text_gt(gt_idx_hum, vocab, tokenized=False)
            text_rom = cap_to_text_gt(gt_idx_rom, vocab, tokenized=False)

            caption_fac = self.tokenizer(text_fact, 
                                padding='max_length', max_length = 25, truncation=True,
                                    return_tensors="pt") 
            caption_hum = self.tokenizer(text_hum, 
                                padding='max_length', max_length = 25, truncation=True,
                                    return_tensors="pt") 
            caption_rom = self.tokenizer(text_rom, 
                                padding='max_length', max_length = 25, truncation=True,
                                    return_tensors="pt")

            mask_fac.append(caption_fac['attention_mask'].to(device))
            mask_hum.append(caption_hum['attention_mask'].to(device))
            mask_rom.append(caption_rom['attention_mask'].to(device))
            input_id_fac.append(caption_fac['input_ids'].squeeze(1).to(device))
            input_id_hum.append(caption_hum['input_ids'].squeeze(1).to(device))
            input_id_rom.append(caption_rom['input_ids'].squeeze(1).to(device))

        #loop  all the combination with 3 style and and n images when it's the same image label is 1 else 0
        n = len(cap_fac)
        for i in range(n):
            j = (i + 1) % n
            gt = 'same img'
            label = torch.tensor([self.labels[gt]]).to(device)
            input_id1 = input_id_fac[i]
            mask1 = mask_fac[i]
            input_id2 = input_id_hum[i]
            mask2 = mask_hum[i]
            pred = self.forward(input_id1, mask1, input_id2, mask2)
            pred_label_int = 1 if pred > 0.0 else 0
            pred_label = torch.tensor([pred_label_int]).to(device)
            if pred_label == label:
                total_acc += 1 
            total_count += 1
            pred = pred.to(torch.float32)
            label = label.to(torch.float32)
            loss +=  F.mse_loss(pred, label) 

            gt = 'diff img'
            label = torch.tensor([self.labels[gt]]).to(device)
            style_2 = random.randint(0,2)
            if style_2 == 0:
                input_id2 = input_id_fac[j]
                mask2 = mask_fac[j]
            elif style_2 == 1:
                input_id2 = input_id_hum[j] 
                mask2 = mask_hum[j] 
            else:
                input_id2 = input_id_rom[j] 
                mask2 = mask_rom[j]
            pred_label_int = 1 if pred > 0.0 else 0
            pred_label = torch.tensor([pred_label_int]).to(device)
            if pred_label == label:
                total_acc += 1 
            total_count += 1
            label = label.to(torch.float32)
            loss +=  F.mse_loss(pred, label) 

            gt = 'same img'
            label = torch.tensor([self.labels[gt]]).to(device)
            input_id2 = input_id_rom[i]
            mask2 = mask_rom[i]
            pred = self.forward(input_id1, mask1, input_id2, mask2)
            pred_label_int = 1 if pred > 0.0 else 0
            pred_label = torch.tensor([pred_label_int]).to(device)
            if pred_label == label:
                total_acc += 1 
            total_count += 1
            label = label.to(torch.float32)
            loss +=  F.mse_loss(pred, label) 

            gt = 'diff img'
            label = torch.tensor([self.labels[gt]]).to(device)
            style_2 = random.randint(0,2)
            if style_2 == 0:
                input_id1 = input_id_fac[j]
                mask1 = mask_fac[j]
            elif style_2 == 1:
                input_id1 = input_id_hum[j] 
                mask1 = mask_hum[j] 
            else:
                input_id1 = input_id_rom[j] 
                mask1 = mask_rom[j]
            pred_label_int = 1 if pred > 0.0 else 0
            pred_label = torch.tensor([pred_label_int]).to(device)
            if pred_label == label:
                total_acc += 1 
            total_count += 1
            label = label.to(torch.float32)
            loss +=  F.mse_loss(pred, label) 

            gt = 'same img'
            label = torch.tensor([self.labels[gt]]).to(device)
            input_id1 = input_id_hum[i]
            mask1 = mask_hum[i]
            pred = self.forward(input_id1, mask1, input_id2, mask2)
            pred_label_int = 1 if pred > 0.0 else 0
            pred_label = torch.tensor([pred_label_int]).to(device)
            if pred_label == label:
                total_acc += 1 
            total_count += 1
            label = label.to(torch.float32)
            loss +=  F.mse_loss(pred, label) 

            gt = 'diff img'
            label = torch.tensor([self.labels[gt]]).to(device)
            style_2 = random.randint(0,2)
            if style_2 == 0:
                input_id2 = input_id_fac[j]
                mask2 = mask_fac[j]
            elif style_2 == 1:
                input_id2 = input_id_hum[j] 
                mask2 = mask_hum[j] 
            else:
                input_id2 = input_id_rom[j] 
                mask2 = mask_rom[j]
            pred_label_int = 1 if pred > 0.0 else 0
            pred_label = torch.tensor([pred_label_int]).to(device)
            if pred_label == label:
                total_acc += 1 
            total_count += 1
            label = label.to(torch.float32)
            loss +=  F.mse_loss(pred, label) 

        total_acc /= total_count
        self.log('val_loss', loss)
        self.log('val accuracy', total_acc)

if __name__ == "__main__":
    img_path = "data/flickr7k_images"
    cap_path = "data/factual_train.txt"
    cap_path_humor = "data/humor/funny_train.txt"
    cap_path_romantic = "data/romantic/romantic_train.txt"
    glove_path = "/cortex/users/cohenza4/glove.6B.200d.txt"
    gru_path = "/cortex/users/cohenza4/checkpoint_gru/small_factual/epoch=43-step=1892.ckpt"
    save_path = "/cortex/users/cohenza4/checkpoint/essence_classifier/"
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
                              shuffle=False,  collate_fn=flickr_collate_fn_essence)
    val_loader = DataLoader(val_data, batch_size=32, num_workers=12,
                            shuffle=False,  collate_fn=flickr_collate_fn_essence)
    # model
    model = BertEssenceClassifer(vocab, lr=1e-3)
    print('Loading GloVe Embedding')
    print(model)
    wandb_logger = WandbLogger(save_dir='/cortex/users/cohenza4')
    lr_monitor_callback = pl.callbacks.LearningRateMonitor()
    checkpoint_callback = ModelCheckpoint(dirpath=save_path, monitor="val_loss", save_top_k=1)
    print('Starting Training')
    trainer = pl.Trainer(gpus=[7], num_nodes=1, precision=32,
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
   