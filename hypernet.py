
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils import data
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torchvision
from torchvision import transforms
from models.decoderlstm import DecoderRNN, DecoderGRU
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
    def __init__(self, embed_size, hidden_size, vocab_size, vocab, num_layers=1, type='gru', lr=1e-6):
        super().__init__()

        self.hparams['vocab_size'] = vocab_size
        self.hparams['embed_size'] = embed_size
        self.hparams['hidden_size'] = hidden_size
        self.vocab = vocab
        self.hparams['lr'] = lr
        self.hparams['num_layers'] = num_layers
        self.teacher_forcing_proba = 1.0
        self.metrics = ['bleu', 'meteor', 'rouge']
        self.metrics = [datasets.load_metric(name) for name in self.metrics]

        # create image encoder
        self.image_encoder = torchvision.models.resnet101(pretrained=True)
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        num_ftrs = self.image_encoder.fc.in_features
        
        self.image_encoder.fc = nn.Linear(num_ftrs, embed_size)
        for param in self.image_encoder.fc.parameters():
            param.requires_grad = True

        if type == 'gru':
            self.captioner = DecoderGRU(embed_size, hidden_size, vocab_size, num_layers=num_layers, dropout=False)
        else:    
            self.captioner = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers=num_layers)
        # create HN base
        self.hn_base = nn.Sequential(
            nn.Linear(embed_size, 4*embed_size),
            nn.LeakyReLU(),
            nn.Linear(4*embed_size, 8*embed_size),
            nn.LeakyReLU()
        )
        self.hn_heads = []
        for name, W in self.captioner.named_parameters():
            if name=='embed.weight':
                continue
            if name=='fc_out.weight':
                continue
            if name=='fc_out.bias':
                continue

            w_size = len(W.flatten())
            if w_size < 8*embed_size:
                self.hn_heads.append(nn.Sequential(
                    nn.Linear(8*embed_size, w_size),
                    nn.LeakyReLU(),
                    nn.Linear(w_size, w_size)
                ))
            else:
                if w_size // 8 < 8*embed_size:
                    self.hn_heads.append(nn.Sequential(
                        nn.Linear(8*embed_size, 8*embed_size),
                        nn.LeakyReLU(),
                        nn.Linear(8*embed_size, w_size)
                    ))
                else:
                    self.hn_heads.append(nn.Sequential(
                        nn.Linear(8*embed_size, w_size//8),
                        nn.LeakyReLU(),
                        nn.Linear(w_size//8, w_size)
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
        flip_parameters_to_tensors(self.captioner)
        set_all_parameters(self.captioner, heads_out.reshape(1,-1))
        return self.captioner

    def configure_optimizers(self):
        params = list(self.hn_heads.parameters())
        params.extend(list(self.hn_base.parameters()))
        params.extend(list(self.captioner.embed.parameters()))
        params.extend(list(self.image_encoder.fc.parameters()))
        optimizer = torch.optim.Adam(params, lr=self.hparams['lr'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, cooldown=2)
        return [optimizer], [{'scheduler': scheduler, 'monitor': 'val_loss'}]


    def training_step(self, train_batch, batch_idx):
        imgs, (style, (caps, lengths)) = train_batch
        style = torch.tensor([vocab(style)])
        style = style.type(torch.LongTensor)
        style = style.to(self.device)
        style_embed = self.captioner.embed(style)

        captioner = self.forward(style_embed)
        img_feats = self.image_encoder(imgs.float())
        teacher_forcing = False

        if np.random.binomial(1,self.teacher_forcing_proba):
            teacher_forcing = True

        caps_pred = self.captioner(img_feats, caps.long(), teacher_forcing)

        bleu, meteor, rouge = metric_score(caps, caps_pred, vocab, self.metrics)

        #caps = nn.functional.one_hot(caps.to(torch.int64), num_classes=self.voc_size).float()
        loss = F.cross_entropy(caps_pred.view(-1, self.hparams['vocab_size']), caps.view(-1).long())

        self.log('train_loss', loss)
        self.log('teacher train proba', self.teacher_forcing_proba)
        if self.teacher_forcing_proba > 0.25: 
            self.teacher_forcing_proba = self.teacher_forcing_proba * 0.9995
        return loss

    def validation_step(self, val_batch, batch_idx):
        imgs, (style, (caps, lengths)) = val_batch

        style = torch.tensor([vocab(style)])
        style = style.type(torch.LongTensor)
        style = style.to(self.device)
        style_embed = self.captioner.embed(style)

        captioner = self.forward(style_embed)
        img_feats = self.image_encoder(imgs.float())
        caps_pred = captioner(img_feats, caps.long(), teacher_forcing=False)

        # caps = nn.functional.one_hot(caps.to(torch.int64), num_classes=self.voc_size).float()
        loss = F.cross_entropy(caps_pred.view(-1, self.hparams['vocab_size']),
                               caps.view(-1).long())                           
        self.log('val_loss', loss, sync_dist=True)

        bleu, meteor, rouge = metric_score(caps, caps_pred, vocab, self.metrics)
        self.log('val_loss', loss, sync_dist=True)
        self.log('meteor', meteor)
        self.log('bleu', bleu)
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
    glove_path = "/cortex/users/algiser/glove.6B.200d.txt"
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

    train_loader = DataLoader(train_data, batch_size=64, num_workers=16,
                            shuffle=False, collate_fn=lambda x: flickr_collate_style(x, 'factual'))
    val_loader = DataLoader(val_data, batch_size=64, num_workers=16,
                            shuffle=False, collate_fn=lambda x: flickr_collate_style(x, 'factual'))
    # model
    model = HyperNet(200, 150, len(vocab), vocab, 2, type='gru', lr=0.5)
    print(model)
    print('Loading GloVe Embedding')
    model.load_glove_emb(glove_path)

    wandb_logger = WandbLogger(save_dir='/cortex/users/algiser')
    lr_monitor_callback = pl.callbacks.LearningRateMonitor()
    checkpoint_callback = ModelCheckpoint(dirpath='/cortex/users/algiser/checkpoint_hn/', monitor="val_loss")
    print('Starting Training')
    trainer = pl.Trainer(gpus=[1], num_nodes=1, precision=32,
                         logger=wandb_logger,
                         check_val_every_n_epoch=1,
                         default_root_dir='/cortex/users/algiser/checkpoints',
                         log_every_n_steps=20,
                         auto_lr_find=True,
                         callbacks=[lr_monitor_callback, checkpoint_callback],
                         #accelerator='ddp',
                         )
    trainer.tune(model, train_loader, val_loader)
    trainer.fit(model, train_loader, val_loader)
