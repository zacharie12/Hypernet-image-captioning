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
from style_classifier_all import BertClassifer
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import copy
from data_loader_combine import get_dataset, ConcatDataset, combine_collate_fn, collate_fn_test
import build_vocab
from build_vocab import Vocab
import pickle
import random
from pytorch_lightning.loggers import WandbLogger   
from utils import set_all_parameters, flip_parameters_to_tensors, WordVectorLoader, cap_to_text, cap_to_text_gt, metric_score, metric_score_test, get_domain_list, get_hist_embedding, tfidf_hist, get_jsd_tsne
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



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class HyperNetCC(pl.LightningModule):
    def __init__(self, feature_size, embed_size, hidden_size, vocab_size, vocab, list_domain, lr=1e-6, mixup=False, alpha=0.3, hyper_emb=10, embedding = 'one hot', n_tsne=2, zero_shot=True, style_classifier_path=""):
        super().__init__()
        train_file = 'data/train_cap_100.txt'
        all_cap_file = "data/all_caption.txt"
        self.hparams['feature_size'] = feature_size
        self.hparams['vocab_size'] = vocab_size
        self.hparams['embed_size'] = embed_size
        self.hparams['hidden_size'] = hidden_size
        self.vocab = vocab
        self.hparams['lr'] = lr
        self.teacher_forcing_proba = 0.0
        self.beam_size = 3
        self.mixup = mixup
        self.alpha = alpha
        self.embedding = embedding
        self.metrics = ['bleu', 'meteor', 'rouge']
        self.metrics = [datasets.load_metric(name) for name in self.metrics]
        self.dict_domain = {} 
        self.list_domain = list_domain   
        self.epoch = 0
        if embedding == 'histograme tfidf':
            if zero_shot:
                self.dict_domain = tfidf_hist(all_cap_file, vocab, list_domain)
            else:
                self.dict_domain = tfidf_hist(train_file, vocab, list_domain)
        elif embedding == 'histograme log':
            if zero_shot:
                self.dict_domain = get_hist_embedding(all_cap_file, vocab, list_domain)
            else:
                self.dict_domain = get_hist_embedding(train_file, vocab, list_domain)
        elif embedding == 'histograme':
            if zero_shot:
                self.dict_domain = get_hist_embedding(all_cap_file, vocab, list_domain, False)
            else:
                self.dict_domain = get_hist_embedding(train_file, vocab, list_domain, False)
        elif embedding == 'JSD':
            if zero_shot:
                self.dict_domain = get_jsd_tsne(all_cap_file, vocab, list_domain, len(list_domain), n_tsne)
            else:
                self.dict_domain = get_jsd_tsne(train_file, vocab, list_domain, len(list_domain), n_tsne)
        else:
            for i in range(len(list_domain)):
                self.dict_domain[list_domain[i].replace("\n", '')] = i
        if embedding == 'one hot':
            x = torch.tensor(list(self.dict_domain.values()))
            self.embed = torch.nn.functional.one_hot(x, len(self.dict_domain))
            self.hyper_emb = len(self.dict_domain)
        elif embedding == 'embedding':
            self.embed = nn.Embedding(len(self.dict_domain), hyper_emb)
            self.hyper_emb = hyper_emb
        elif embedding == 'histograme log' or embedding == 'histograme tfidf' or embedding == 'histograme':
            self.embed = nn.Sequential(
            nn.Linear(len(vocab)+1, hyper_emb*4),
            nn.LeakyReLU(),
            nn.Linear( hyper_emb*4, hyper_emb),
            nn.LeakyReLU()
        )
            self.hyper_emb = hyper_emb
        elif embedding == 'JSD':
            self.embed = nn.Sequential(
            nn.Linear(n_tsne, hyper_emb),
            nn.LeakyReLU()
        )
            self.hyper_emb = hyper_emb
        self.image_encoder = EncoderCNN()
        self.hypernet = HyperNet(feature_size, embed_size, hidden_size, vocab_size, vocab, num_layers=1, lr=lr, mixup=mixup, alpha=alpha, cc=True, hyper_emb=self.hyper_emb )
        if mixup:
            self.style_classifier = BertClassifer(vocab,  num_class=4)
            self.style_classifier.load_from_checkpoint(checkpoint_path=style_classifier_path, vocab=vocab)

    def configure_optimizers(self):
        params = list(self.hypernet.hn_heads.parameters())
        if not self.embedding == 'one hot':
            params.extend(list(self.embed.parameters()))
        params.extend(list(self.hypernet.hn_base.parameters()))
        params.extend(list(self.hypernet.captioner.feature_fc.parameters()))
        params.extend(list(self.hypernet.captioner.embed.parameters()))
        params.extend(list(self.hypernet.captioner.fc.parameters()))
        params.extend(list(self.hypernet.captioner.attention.parameters()))
        params.extend(list(self.hypernet.captioner.init_h.parameters()))
        optimizer = torch.optim.Adam(params, lr=self.hparams['lr'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, cooldown=2, factor=0.5)
        return [optimizer], [{'scheduler': scheduler, 'monitor': 'val_loss with TF', 'interval': 'epoch'}]


    def load_glove_emb(self, emb_path):
        loader = WordVectorLoader(self.hparams['embed_size'])
        loader.load_glove(emb_path)
        emb_mat = loader.generate_embedding_matrix(self.vocab.w2i,
                                                   self.vocab.ix-1, 'norm')
        emb_mat = torch.FloatTensor(emb_mat)
        self.hypernet.captioner.embed = self.hypernet.captioner.embed.from_pretrained(emb_mat,
                                                                    freeze=False)

    def training_step(self, train_batch, batch_idx):
        imgs, caps, lengths, domains = train_batch
        domain = domains[0]
        if self.embedding == "embedding":
            domain = torch.tensor(self.dict_domain[domain])
            domain = domain.type(torch.LongTensor).to(self.device)
            style_embed = self.embed(domain)
        elif self.embedding == 'one hot':
            domain = self.dict_domain[domain]
            style_embed = torch.tensor(self.embed[domain])
            style_embed = style_embed.type(torch.FloatTensor).to(self.device)
        else:
            domain = self.dict_domain[domain]
            domain =  torch.tensor(domain)
            domain = domain.type(torch.FloatTensor).to(self.device)
            style_embed = self.embed(domain)
        captioner = self.hypernet.forward(style_embed)
        img_feats = self.hypernet.image_encoder(imgs.float()) 
        caps_pred, _ = captioner(img_feats, caps.long(), self.teacher_forcing_proba)
        loss =  F.cross_entropy(caps_pred.view(-1, self.hparams['vocab_size']), caps.view(-1).long(), ignore_index=self.vocab.w2i['<pad>'])       
        bleu1, bleu2, bleu3, bleu4, meteor, rouge, cider = metric_score(caps, caps_pred, self.vocab, self.metrics)

        self.log('LR', self.hparams['lr'])
        self.log('train_loss', loss)
        self.log('meteor train', meteor)
        self.log('bleu 1 train', bleu1)
        self.log('bleu 2 train', bleu2)
        self.log('bleu 3 train', bleu3)
        self.log('bleu 4 train', bleu4)
        self.log('rouge train', rouge)
        self.log('cider train', cider)


        if self.mixup:
            style_list = ["f", "h", "r", "CC"]
            coeff = random.random()
            no_style1 = random.choice(style_list)
            style_list.remove(no_style1)
            no_style2 = random.choice(style_list)
            style_list.remove(no_style2)
            style1, style2 = style_list[0], style_list[1]
            label1, label2 = torch.tensor([self.style_classifier.labels[style1]]).to(device), torch.tensor([self.style_classifier.labels[style2]]).to(device)
            if style1 == "CC":
                style1 = self.list_domain[random.randint(0,99)].replace("\n", '') 
            if style2 == "CC":
                style2 = self.list_domain[random.randint(0,99)].replace("\n", '') 

            if self.embedding == "embedding":
                style1 = torch.tensor(self.dict_domain[style1])
                style1 = domain.type(torch.LongTensor).to(self.device)
                style_embed1 = self.embed(style1)
                style2 = torch.tensor(self.dict_domain[style2])
                style2 = domain.type(torch.LongTensor).to(self.device)
                style_embed2 = self.embed(style2)
            elif self.embedding == 'one hot':
                style1 = self.dict_domain[style1]
                style_embed1 = torch.tensor(self.embed[style1])
                style_embed1 = style_embed1.type(torch.FloatTensor).to(self.device)
                style2 = self.dict_domain[style2]
                style_embed2 = torch.tensor(self.embed[style2])
                style_embed2 = style_embed2.type(torch.FloatTensor).to(self.device)
            else:
                style1 = self.dict_domain[style1]
                style1 =  torch.tensor(style1)
                style1 = style1.type(torch.FloatTensor).to(self.device)
                style_embed1 = self.embed(style1)
                style2 = self.dict_domain[style2]
                style2 =  torch.tensor(style2)
                style2 = style2.type(torch.FloatTensor).to(self.device)
                style_embed2 = self.embed(style2)

            mixup_style = coeff*style_embed1 + (1-coeff)*style_embed2 
            captioner_mixup = self.hypernet.forward(mixup_style)
            caps_pred_mixup, _ = captioner_mixup(img_feats, caps.long(), 1.0)
            for i in range(len(caps)):
                gt_idx = torch.squeeze(caps_pred_mixup[i])
                text = cap_to_text(gt_idx, self.vocab, tokenized=False)
            caption = self.style_classifier.tokenizer(text, 
                                padding='max_length', max_length = 25, truncation=True,
                                    return_tensors="pt") 
            mask = caption['attention_mask'].to(device)
            input_id = caption['input_ids'].squeeze(1).to(device)
            style_pred = self.style_classifier(input_id, mask)            
            if no_style1 == "f":
                target = torch.tensor([0.0, coeff, 0.0, 0.0]).to(device)
            elif no_style1 == "h":
                target = torch.tensor([coeff, 0.0, 0.0, 0.0]).to(device)
            elif no_style1 == "r":
                target = torch.tensor([coeff,0.0, 0.0, 0.0]).to(device)
            else:
                target = torch.tensor([0.0, 0.0, 0.0, coeff]).to(device)
            if no_style2 == "f":
                target += torch.tensor([0.0, 1-coeff, 0.0, 0.0]).to(device)
            elif no_style2 == "h":
                target += torch.tensor([1-coeff, 0.0, 0.0, 0.0]).to(device)
            elif no_style2 == "r":
                target += torch.tensor([1-coeff,0.0, 0.0, 0.0]).to(device)
            else:
                target += torch.tensor([0.0, 0.0, 0.0, 1-coeff]).to(device)
            style_loss = F.mse_loss(style_pred, target)  

            if self.mixup:
                loss = self.alpha * loss + (1-self.alpha) * style_loss
                self.log('style loss', style_loss)
        return loss


    def validation_step(self, val_batch, batch_idx):      
        imgs, caps, lengths, domains = val_batch
        fliker_domain = ['f', 'r', 'h']
        domain = domains[0]
        style =domain
        if self.embedding == "embedding":
            domain = torch.tensor(self.dict_domain[domain])
            domain = domain.type(torch.LongTensor).to(self.device)
            style_embed = self.embed(domain)
        elif self.embedding == 'one hot':
            domain = self.dict_domain[domain]
            style_embed = torch.tensor(self.embed[domain])
            style_embed = style_embed.type(torch.FloatTensor).to(self.device)
        else:
            domain = self.dict_domain[domain]
            domain =  torch.tensor(domain)
            domain = domain.type(torch.FloatTensor).to(self.device)
            style_embed = self.embed(domain)
        captioner = self.hypernet.forward(style_embed)
        img_feats = self.hypernet.image_encoder(imgs.float())
        caps_pred_teacher_forcing, _ = captioner(img_feats, caps.long(), self.teacher_forcing_proba)
        caps_pred, _ = captioner(img_feats, caps.long(), 1.0)

        loss = F.cross_entropy(caps_pred.view(-1, self.hparams['vocab_size']),
                               caps.view(-1).long(), ignore_index=self.vocab.w2i['<pad>'])

        loss_tf = F.cross_entropy(caps_pred_teacher_forcing.view(-1, self.hparams['vocab_size']),
                               caps.view(-1).long(), ignore_index=self.vocab.w2i['<pad>'])
        
        bleu1, bleu2, bleu3, bleu4, meteor, rouge, cider = metric_score(caps, caps_pred, self.vocab, self.metrics)

        if style in fliker_domain:
            self.log('val_loss fliker', loss, sync_dist=True)
            self.log('meteor val fliker', meteor)
            self.log('bleu 1 val fliker', bleu1)
            self.log('bleu 2 val fliker', bleu2)
            self.log('bleu 3 val fliker', bleu3)
            self.log('bleu 4 val fliker', bleu4)
            self.log('rouge val fliker', rouge)
            self.log('cider val fliker', cider)
        
        else:
            self.log('val_loss CC', loss, sync_dist=True)
            self.log('meteor val CC', meteor)
            self.log('bleu 1 val CC', bleu1)
            self.log('bleu 2 val CC', bleu2)
            self.log('bleu 3 val CC', bleu3)
            self.log('bleu 4 val CC', bleu4)
            self.log('rouge val CC', rouge)
            self.log('cider val CC', cider)

        self.log('val_loss with TF', loss_tf, sync_dist=True)



    def test_step(self, test_batch, batch_idx):
        imgs, caps, lengths, domains = test_batch
        fliker_domain = ['f', 'r', 'h']
        domain = domains[0]
        style = domain
        if self.embedding == "embedding":
            domain = torch.tensor(self.dict_domain[domain])
            domain = domain.type(torch.LongTensor).to(self.device)
            style_embed = self.embed(domain)
        elif self.embedding == 'one hot':
            domain = self.dict_domain[domain]
            style_embed = torch.tensor(self.embed[domain])
            style_embed = style_embed.type(torch.FloatTensor).to(self.device)
        else:
            domain = self.dict_domain[domain]
            domain =  torch.tensor(domain)
            domain = domain.type(torch.FloatTensor).to(self.device)
            style_embed = self.embed(domain)
        captioner = self.hypernet.forward(style_embed)
        img_feats = self.hypernet.image_encoder(imgs.float())
        caps_pred, _ = captioner(img_feats, caps.long(), 0.5)
 
        bleu1, bleu2, bleu3, bleu4, meteor, rouge, cider = metric_score(caps, caps_pred, self.vocab, self.metrics)

        if style in fliker_domain:
            self.log('meteor test {}'.format(style), meteor)
            self.log('bleu 1 test {}'.format(style), bleu1)
            self.log('bleu 2 test {}'.format(style), bleu2)
            self.log('bleu 3 test {}'.format(style), bleu3)
            self.log('bleu 4 test {}'.format(style), bleu4)
            self.log('rouge test {}'.format(style), rouge)
            self.log('cider test {}'.format(style), cider)
        else:
            self.log('meteor test CC', meteor)
            self.log('bleu 1 test CC', bleu1)
            self.log('bleu 2 test CC', bleu2)
            self.log('bleu 3 test CC', bleu3)
            self.log('bleu 4 test CC', bleu4)
            self.log('rouge test CC', rouge)
            self.log('cider test CC', cider)

if __name__ == "__main__":
    glove_path = "/cortex/users/cohenza4/glove.6B.200d.txt"
    img_dir_fliker = 'data/flickr7k_images/'
    img_dir_CC_train = 'data/200_conceptual_images_train/'
    img_dir_CC_val_test = 'data/200_conceptual_images_val/'
    caption_CC_train = 'data/train_cap_100.txt'
    caption_CC_val = 'data/val_cap_100.txt'
    caption_CC_test = 'data/test_cap_100.txt'
    caption_fac = 'data/fac_cap_train.txt'
    caption_hum = 'data/humor_cap_train.txt'
    caption_rom = 'data/rom_cap_train.txt'
    caption_fac_test = 'data/fac_cap_test.txt'
    caption_hum_test = 'data/humor_cap_test.txt'
    caption_rom_test = 'data/rom_cap_test.txt'
    zero_shot_captions = 'data/one_shot_captions.txt'
    zero_shot_images = 'data/one_shot_images/'
    style_classifier_path = '/cortex/users/cohenza4/checkpoint/style_classifier/epoch=29-step=4889.ckpt'
    save_path = "/cortex/users/cohenza4/checkpoint/HN_combine_style_loss/one_hot/"
    with open("data/vocab.pkl", 'rb') as f:
        vocab = pickle.load(f)

    data_fac = get_dataset(img_dir_fliker, caption_fac, vocab, "Fliker factual")
    data_hum = get_dataset(img_dir_fliker, caption_hum, vocab, "Fliker style")
    data_rom = get_dataset(img_dir_fliker, caption_rom, vocab, "Fliker style")
    train_data_CC = get_dataset(img_dir_CC_train, caption_CC_train, vocab)
    val_data_CC = get_dataset(img_dir_CC_val_test, caption_CC_val, vocab)
    
    
    lengths_fac = [int(len(data_fac)*0.9), len(data_fac) - int(len(data_fac)*0.9)]
    lengths_hum = [int(len(data_hum)*0.9), len(data_hum) - int(len(data_hum)*0.9)]
    lengths_rom = [int(len(data_rom)*0.9), len(data_rom) - int(len(data_rom)*0.9)]

    train_data_fac, val_data_fac = torch.utils.data.random_split(data_fac, lengths_fac)
    train_data_hum, val_data_hum = torch.utils.data.random_split(data_hum, lengths_hum)
    train_data_rom, val_data_rom = torch.utils.data.random_split(data_rom, lengths_rom)

    test_data_fac = get_dataset(img_dir_fliker, caption_fac_test, vocab, "Fliker factual")
    test_data_hum = get_dataset(img_dir_fliker, caption_hum_test, vocab, "Fliker style")
    test_data_rom = get_dataset(img_dir_fliker, caption_rom_test, vocab, "Fliker style")
    test_data_CC = get_dataset(img_dir_CC_val_test, caption_CC_test, vocab)
    test_data_CC_zeroshot = get_dataset(zero_shot_images, zero_shot_captions, vocab)

    train_data_concat = ConcatDataset(train_data_fac, train_data_hum, train_data_rom, train_data_CC)
    val_data_concat = ConcatDataset(val_data_fac, val_data_hum, val_data_rom, val_data_CC)

    train_loader = DataLoader(train_data_concat, batch_size=32, num_workers=2, shuffle=False, collate_fn= combine_collate_fn)
    val_loader = DataLoader(val_data_concat, batch_size=10, num_workers=2, shuffle=False, collate_fn= combine_collate_fn)

    test_data_concat = ConcatDataset(test_data_fac, test_data_hum, test_data_rom, test_data_CC, test_data_CC_zeroshot)

    test_loader_fac = DataLoader(test_data_concat, batch_size=32, num_workers=2, shuffle=False,  collate_fn=lambda x: collate_fn_test(x, 'factual'))
    test_loader_hum = DataLoader(test_data_concat, batch_size=32, num_workers=2, shuffle=False,  collate_fn=lambda x: collate_fn_test(x, 'humour'))
    test_loader_rom = DataLoader(test_data_concat, batch_size=32, num_workers=2, shuffle=False,  collate_fn=lambda x: collate_fn_test(x, 'romantic'))
    test_loader_CC = DataLoader(test_data_concat, batch_size=10, num_workers=2, shuffle=False,  collate_fn=lambda x: collate_fn_test(x, 'CC'))
    test_loader_CC_zeroshot = DataLoader(test_data_concat, batch_size=20, num_workers=2, shuffle=False,  collate_fn=lambda x: collate_fn_test(x, 'CC'))

    list_domain_cc = get_domain_list(caption_CC_train, zero_shot_captions)
    list_fliker = ['f', 'r', 'h']
    list_domain = list_domain_cc + list_fliker

    #'histograme' 'histograme log' 'histograme tfidf' 'JSD', "embedding", "one hot"
    domain_emb = 'one hot'
    style_loss = True

    model = HyperNetCC(200, 200, 200, len(vocab), vocab, list_domain, 0.001, style_loss, 0.3, 10, domain_emb, style_classifier_path=style_classifier_path)

    print('Loading GloVe Embedding')
    model.load_glove_emb(glove_path)
    print(model)
    wandb_logger = WandbLogger(save_dir='/cortex/users/cohenza4')
    lr_monitor_callback = pl.callbacks.LearningRateMonitor()
    checkpoint_callback = ModelCheckpoint(dirpath=save_path, monitor="val_loss with TF", save_top_k=1)
    print('Starting Training')
    trainer = pl.Trainer(gpus=[5], num_nodes=1, precision=32,
                         logger=wandb_logger,
                         #overfit_batches = 1,
                         check_val_every_n_epoch=1,
                         default_root_dir=save_path,
                         log_every_n_steps=20,
                         auto_lr_find=False,
                         callbacks=[lr_monitor_callback, checkpoint_callback],
                         #accelerator='ddp',
                         max_epochs=55,
                         gradient_clip_val=5.,
                         )
    trainer.tune(model, train_loader, val_loader)
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader_fac)
    trainer.test(model, test_loader_hum)
    trainer.test(model, test_loader_rom)
    trainer.test(model, test_loader_CC)
    trainer.test(model, test_loader_CC_zeroshot)