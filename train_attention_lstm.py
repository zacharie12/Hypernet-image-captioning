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
from models.decoderlstm import DecoderWithAttention
from models.encoder import Encoder
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
from torch.nn.utils.rnn import pack_padded_sequence

class CaptionAttentionLstm(pl.LightningModule):
    def __init__(self, encoded_image_size, attention_dim, embed_dim, decoder_dim, vocab_size, vocab, encoder_dim=2048,
                 dropout=0.5, lr=1e-6):
        super().__init__()

        self.hparams['encoded_image_size'] = encoded_image_size
        self.hparams['attention_dim'] = attention_dim
        self.hparams['embed_dim'] = embed_dim
        self.hparams['decoder_dim'] = decoder_dim
        self.hparams['vocab_size'] = vocab_size
        self.hparams['encoder_dim'] = encoder_dim
        self.hparams['dropout'] = dropout
        self.vocab = vocab
        self.hparams['lr'] = lr   
        # create image encoder and decodeur
        self.encoder = Encoder(encoded_image_size=encoded_image_size)
        self.decoder = DecoderWithAttention(attention_dim, embed_dim, decoder_dim, vocab_size,
                                            encoder_dim, dropout)
        self.metrics = ['bleu', 'meteor', 'rouge']
        self.metrics = [datasets.load_metric(name) for name in self.metrics]


    def load_glove_emb(self, emb_path):
        loader = WordVectorLoader(self.hparams['embed_dim'])
        loader.load_glove(emb_path)
        emb_mat = loader.generate_embedding_matrix(self.vocab.w2i,
                                                   self.vocab.ix-1, 'norm')
        emb_mat = torch.FloatTensor(emb_mat)

        self.decoder.embedding  = self.decoder.embedding.from_pretrained(emb_mat,
                                                                    freeze=True)
    
    def forward(self, images, encoded_captions, caption_lengths):
        """
        :param images: [b, 3, h, w]
        :param encoded_captions: [b, max_len]
        :param caption_lengths: [b,]
        :return:
        """
        encoder_out = self.encoder(images)
        decoder_out = self.decoder(encoder_out, encoded_captions, caption_lengths.unsqueeze(1))
        return decoder_out

    def sample(self, images, startseq_idx, endseq_idx=-1, max_len=40, return_alpha=False):
        encoder_out = self.encoder(images)
        return self.decoder.sample(encoder_out=encoder_out, startseq_idx=startseq_idx, max_len=max_len,
                                   return_alpha=return_alpha)        
    def sample_val(self, images, startseq_idx, caption_lengths, encoded_captions, endseq_idx=-1, return_alpha=False):
        encoder_out = self.encoder(images)
        return self.decoder.sample_val(encoder_out=encoder_out, startseq_idx=startseq_idx, caption_lengths=caption_lengths.unsqueeze(1),
                                   encoded_captions=encoded_captions, return_alpha=return_alpha)  

    def configure_optimizers(self):
        params = list(self.encoder.parameters())
        params.extend(list(self.decoder.parameters()))
        optimizer = torch.optim.Adam(params, lr=self.hparams['lr'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)
        return [optimizer], [{'scheduler': scheduler, 'monitor': 'val_loss'}]
        #return optimizer
  
    def training_step(self, train_batch, batch_idx):
        imgs, (style, (caps, lengths)) = train_batch
        scores, caps_sorted, decode_lengths, alphas, sort_ind = self.forward(imgs, caps, lengths)
        targets = caps_sorted[:, 1:]
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)[0]
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)[0]
        loss =  F.cross_entropy(scores, targets, ignore_index=self.vocab.w2i['<pad>']) 
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        imgs, (style, (caps, lengths)) = val_batch

        scores, decode_lengths, caps_sorted, outputs  = self.sample_val(imgs, startseq_idx=self.vocab.w2i['<s>'], caption_lengths=lengths, encoded_captions=caps)
        targets = caps_sorted[:, 1:]
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)[0]
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)[0]
        loss =  F.cross_entropy(scores, targets, ignore_index=self.vocab.w2i['<pad>'])
        a, b = caps.size()    
        pred =  scores.view(a, b, len(self.vocab))
        bleu, meteor,_ = metric_score(caps, pred, self.vocab, self.metrics)
        self.log('val_loss', loss)
        """
        outputs  = self.sample(imgs, startseq_idx=self.vocab.w2i['<s>'])
        """


        
if __name__ == "__main__":
    img_path = "data/flickr7k_images"
    cap_path = "data/factual_train.txt"
    cap_path_humor = "data/humor/funny_train.txt"
    cap_path_romantic = "data/romantic/romantic_train.txt"
    glove_path = "/cortex/users/cohenza4/glove.6B.300d.txt"
    gru_path = "/cortex/users/cohenza4/chekpoints/factual_gru/epoch=16-step=1408.ckpt"
    save_model = "/cortex/users/cohenza4/chekpoints/factual_lstm/"
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
    val_loader = DataLoader(val_data, batch_size=128, num_workers=24,
                            shuffle=False, collate_fn=lambda x: flickr_collate_style(x, 'factual'))
    # model
    model = CaptionAttentionLstm(14, 256, 300,256, len(vocab), vocab, lr=0.00275)
    #model.load_glove_emb(glove_path)
    print('Loading GloVe Embedding')
    #print(model)
    wandb_logger = WandbLogger(save_dir='/cortex/users/cohenza4')
    wandb_logger.log_hyperparams(model.hparams)
    lr_monitor_callback = pl.callbacks.LearningRateMonitor()
    checkpoint_callback = ModelCheckpoint(dirpath=save_model, monitor="val_loss")
    print('Starting Training')
    trainer = pl.Trainer(gpus=[2], num_nodes=1, precision=32,
                         logger=wandb_logger,
                         check_val_every_n_epoch=1,
                         #overfit_batches=10,
                         default_root_dir=save_model,
                         log_every_n_steps=20,
                         auto_lr_find=True,
                         callbacks=[lr_monitor_callback, checkpoint_callback],
                         gradient_clip_val=5.,
                         #accelerator='ddp',
                         )                                   
    trainer.tune(model, train_loader, val_loader)
    trainer.fit(model, train_loader, val_loader)