import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils import data
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torchvision
from torchvision import transforms
from models.decoderlstm import DecoderLstm
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
from models.encoder import EncoderLstm   
from utils import set_all_parameters, flip_parameters_to_tensors, WordVectorLoader, cap_to_text, cap_to_text_gt, metric_score
import nltk
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from nltk.translate.meteor_score import single_meteor_score, meteor_score
import datasets


class CaptionLstm(pl.LightningModule):


    def __init__(self, embed_size, hidden_size, vocab_size, vocab, num_layers=1, lr=1e-6):
        super().__init__()

        self.hparams['vocab_size'] = vocab_size
        self.hparams['embed_size'] = embed_size
        self.hparams['hidden_size'] = hidden_size
        self.vocab = vocab
        self.hparams['lr'] = lr
        self.hparams['num_layers'] = num_layers
        self.teacher_forcing_proba = 0.5
    
        # create image encoder
        self.encoder = EncoderLstm(embed_size)
        self.decoder = DecoderLstm(embed_size, hidden_size, vocab_size)

        self.metrics = ['bleu', 'meteor', 'rouge']
        self.metrics = [datasets.load_metric(name) for name in self.metrics]


    def load_glove_emb(self, emb_path):
        loader = WordVectorLoader(self.hparams['embed_size'])
        loader.load_glove(emb_path)
        emb_mat = loader.generate_embedding_matrix(self.vocab.w2i,
                                                   self.vocab.ix-1, 'norm')
        emb_mat = torch.FloatTensor(emb_mat)
        self.decoder.word_embeddings = self.decoder.word_embeddings.from_pretrained(emb_mat,
                                                                    freeze=True)
    
    def forward(self, img, caption_gt):
        features = self.encoder(img).unsqueeze(1)
        output = self.decoder.forward(features, caption_gt)
        return output

    def configure_optimizers(self):
        params = list(self.decoder.parameters()) + list(self.encoder.embed.parameters())
        optimizer = torch.optim.Adam(params, lr=self.hparams['lr'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, cooldown=2)
        return [optimizer], [{'scheduler': scheduler, 'monitor': 'val_loss'}]
        #return optimizer
  
    def training_step(self, train_batch, batch_idx):
        imgs, (style, (caps, lengths)) = train_batch
        caps_pred = self.forward(imgs.float(), caps.long())
        loss = F.cross_entropy(caps_pred.view(-1, self.hparams['vocab_size']), caps.view(-1).long())        
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        imgs, (style, (caps, lengths)) = val_batch
        features = self.encoder(imgs.float()).unsqueeze(1)
        cap_pred = self.decoder.sample(features)

        loss = F.cross_entropy(cap_pred.view(-1, self.hparams['vocab_size']), caps.view(-1).long())        
        bleu, meteor, _ = metric_score(caps, cap_pred, vocab, self.metrics)
        self.log('val_loss', loss, sync_dist=True)
        self.log('meteor', meteor)
        self.log('bleu', bleu)
        #self.log('rouge', rouge)



if __name__ == "__main__":
    img_path = "data/flickr7k_images"
    cap_path = "data/factual_train.txt"
    cap_path_humor = "data/humor/funny_train.txt"
    cap_path_romantic = "data/romantic/romantic_train.txt"
    glove_path = "/cortex/users/cohenza4/glove.6B.300d.txt"
    save_path = '/cortex/users/cohenza4/checkpoints_lstm/'
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

    train_loader = DataLoader(train_data, batch_size=128, num_workers=24,
                              shuffle=False, collate_fn=lambda x: flickr_collate_style(x, 'factual'))
    val_loader = DataLoader(val_data, batch_size=1, num_workers=1,
                            shuffle=False, collate_fn=lambda x: flickr_collate_style(x, 'factual'))
    # model
    model = CaptionLstm(300, 300, len(vocab), vocab, 1, lr=1e-3)
    print('Loading GloVe Embedding')
    #model.load_glove_emb(glove_path)


    wandb_logger = WandbLogger(save_dir='/cortex/users/cohenza4')
    wandb_logger.log_hyperparams(model.hparams)
    lr_monitor_callback = pl.callbacks.LearningRateMonitor()
    checkpoint_callback = ModelCheckpoint(dirpath=save_path, monitor="val_loss", save_top_k=3)
    trainer = pl.Trainer(gpus=[5], num_nodes=1, precision=32,
                         logger=wandb_logger,
                         check_val_every_n_epoch=1,
                         #overfit_batches=5,
                         default_root_dir=save_path,
                         log_every_n_steps=20,
                         auto_lr_find=True,
                         callbacks=[lr_monitor_callback, checkpoint_callback],
                         #max_epochs=10,
                         gradient_clip_val=5.,
                         #accelerator='ddp',
                         )                                
    trainer.tune(model, train_loader, val_loader)
    trainer.fit(model, train_loader, val_loader)