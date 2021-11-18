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
from data_loader import get_dataset, get_styled_dataset, flickr_collate_fn, ConcatDataset, FlickrCombiner, flickr_collate_style
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

class CaptionAttentionGru(pl.LightningModule):
    def __init__(self, features_size, embed_size, hidden_size, vocab_size, vocab, num_layers=1, lr=1e-6):
        super().__init__()

        self.hparams['vocab_size'] = vocab_size
        self.hparams['embed_size'] = embed_size
        self.hparams['hidden_size'] = hidden_size
        self.hparams['features_size'] = features_size
        self.vocab = vocab
        self.hparams['lr'] = lr
        self.hparams['num_layers'] = num_layers
        self.teacher_forcing_proba = 0.0
    
        # create image encoder
        self.image_encoder = EncoderCNN()
            
        self.captioner = AttentionGru(2048, features_size, embed_size, hidden_size, vocab_size, num_layers=num_layers, p=0.0)
        self.metrics = ['bleu', 'meteor', 'rouge']
        self.metrics = [datasets.load_metric(name) for name in self.metrics]


    def load_glove_emb(self, emb_path):
        loader = WordVectorLoader(self.hparams['embed_size'])
        loader.load_glove(emb_path)
        emb_mat = loader.generate_embedding_matrix(self.vocab.w2i,
                                                   self.vocab.ix-1, 'norm')
        emb_mat = torch.FloatTensor(emb_mat)

        self.captioner.embed = self.captioner.embed.from_pretrained(emb_mat,
                                                                    freeze=True)
    def build_html(self, output, out_path):
        doc = dominate.document(title='StyleFlicker7k')
        with doc:
            with table().add(tbody()):
                with tr():
                    with td():
                        p('Image')
                    with td():
                        p('Predicted Factual')
                    with td():
                        p('GT Factual')
                    with td():
                        p('GT Humorous')
                    with td():
                        p('GT Romantic')
            
                for (img_path, fac_pred, fac_gt, hum_gt, rom_gt) in output:
                #for (fac_pred, fac_gt, hum_gt, rom_gt) in output:
                    caps = [fac_pred, fac_gt, hum_gt, rom_gt]
                    with tr():
                        with td():
                           img(src=img_path)
                        for cap in caps:
                            with td():
                                p(cap)

        with open(out_path, 'w') as outfile:
                outfile.write(str(doc))

    def forward(self):
        lstm_cop = self.captioner
        return lstm_cop

    def configure_optimizers(self):
        params = list(self.captioner.parameters())
        #params = list(self.captioner.gru.parameters())
        optimizer = torch.optim.Adam(params, lr=self.hparams['lr'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, cooldown=2, factor=0.5)
        return [optimizer], [{'scheduler': scheduler, 'monitor': 'val_loss with TF', 'interval': 'epoch'}]
        #return optimizer
  
    def training_step(self, train_batch, batch_idx):
        imgs, (style, (caps, lengths)) = train_batch
        img_feats = self.image_encoder(imgs.float()) 
        caps_pred, _ = self.captioner(img_feats, caps.long(), self.teacher_forcing_proba)
        loss =  F.cross_entropy(caps_pred.view(-1, self.hparams['vocab_size']), caps.view(-1).long(), ignore_index=self.vocab.w2i['<pad>'])        
        bleu, meteor, rouge = metric_score(caps, caps_pred, self.vocab, self.metrics)
        self.log('train_loss', loss)
        self.log('LR', self.hparams['lr'])
        self.log('TF', self.teacher_forcing_proba)
        if self.teacher_forcing_proba < 0.75: 
            self.teacher_forcing_proba = self.teacher_forcing_proba * 1.0005
        return loss

    def validation_step(self, val_batch, batch_idx):
        imgs, (style, (caps, lengths)) = val_batch
        img_feats = self.image_encoder(imgs.float())
        caps_pred_teacher_forcing, _ = self.captioner(img_feats, caps.long(), self.teacher_forcing_proba)
        caps_pred, _ = self.captioner(img_feats, caps.long(), 1.0)

        loss = F.cross_entropy(caps_pred.view(-1, self.hparams['vocab_size']),
                               caps.view(-1).long(), ignore_index=self.vocab.w2i['<pad>'])
        
        bleu, meteor, rouge = metric_score(caps, caps_pred, self.vocab, self.metrics)
        
        self.log('val_loss', loss, sync_dist=True)
        self.log('meteor', meteor)
        self.log('bleu', bleu)
        loss_tf = F.cross_entropy(caps_pred_teacher_forcing.view(-1, self.hparams['vocab_size']),
                               caps.view(-1).long(), ignore_index=self.vocab.w2i['<pad>'])
        self.log('val_loss with TF', loss_tf, sync_dist=True)
        '''
        beam = BeamSearch(decoder=self.captioner, features=img_feats, k=3, max_sentence=20)
        sent, score = beam.generate_caption()
        bleu_beam, meteor_beam, _ = metric_score(caps, sent, self.vocab, self.metrics)
        self.log('meteor beam', meteor_beam)
        self.log('bleu beam', bleu_beam)
        '''



    def test_step(self, test_batch, batch_idx):
        imgs, (style, (caps, lengths)) = test_batch
        img_feats = self.image_encoder(imgs.float())
        caps_pred, _ = self.captioner(img_feats, caps.long(), 1.0)
        bleu, meteor, rouge = metric_score(caps, caps_pred, self.vocab, self.metrics)
        self.log(' test meteor', meteor)
        self.log('test bleu', bleu)                       

        beam = BeamSearch(decoder=self.captioner, features=img_feats, k=3, max_sentence=20)
        sent, score = beam.generate_caption().to(device)
        bleu_beam, meteor_beam, _ = metric_score(caps, sent, self.vocab, self.metrics)
        self.log('test meteor beam', meteor_beam)
        self.log('test bleu beam', bleu_beam)

        
        


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
    save_path = "/cortex/users/cohenza4/checkpoint_gru/factual/"
    save_html = '/home/lab/cohenza4/www/StyleFlicker7k_small_gru_factual.html'
    # data
    with open("data/vocab.pkl", 'rb') as f:
        vocab = pickle.load(f)
    print('Prepairing Data')
    orig_dataset = get_dataset(img_path, cap_path, vocab)
    humor_dataset = get_styled_dataset(cap_path_humor, vocab)
    romantic_dataset = get_styled_dataset(cap_path_romantic, vocab)

    data_concat = ConcatDataset(orig_dataset, humor_dataset, romantic_dataset)
    '''
    lengths = [int(len(data_concat)*0.8),
               len(data_concat) - int(len(data_concat)*0.8)]
    train_data, val_data = torch.utils.data.random_split(data_concat, lengths)

   '''
    lengths = [int(len(data_concat)*0.8), int(len(data_concat)*0.1),
               len(data_concat) - (int(len(data_concat)*0.8) + int(len(data_concat)*0.1))]
    train_data, val_data, test_data = torch.utils.data.random_split(data_concat, lengths)

    test_loader = DataLoader(test_data, batch_size=1, num_workers=1,
                            shuffle=False, collate_fn=lambda x: flickr_collate_style(x, 'factual'))

    '''
    orig_dataset_visu = get_dataset(img_path, cap_path, vocab, get_path=True)
    data_concat_visu = ConcatDataset(orig_dataset_visu, humor_dataset, romantic_dataset)
    train_data_visu, val_data_visu = torch.utils.data.random_split(data_concat_visu, lengths)    

    '''

    print("len test", len(test_data))

    train_loader = DataLoader(train_data, batch_size=128, num_workers=24,
                              shuffle=False, collate_fn=lambda x: flickr_collate_style(x, 'factual'))
    val_loader = DataLoader(val_data, batch_size=128, num_workers=24,
                            shuffle=False, collate_fn=lambda x: flickr_collate_style(x, 'factual'))
    # model
    model = CaptionAttentionGru(200, 200, 200, len(vocab), vocab, lr=0.0003)
    load_checkpoint = False
    if load_checkpoint:
        rnn = CaptionAttentionGru(200, 100, 180, len(vocab), vocab)
        rnn = rnn.load_from_checkpoint(checkpoint_path=gru_path, vocab=vocab)
        model.captioner.embed = rnn.captioner.embed
        model.captioner.fc = rnn.captioner.fc
        model.captioner.attention = rnn.captioner.attention
        model.captioner.init_h = rnn.captioner.init_h
    else:
        print('Loading GloVe Embedding')
        model.load_glove_emb(glove_path)
    print(model)
    wandb_logger = WandbLogger(save_dir='/cortex/users/cohenza4')
    wandb_logger.log_hyperparams(model.hparams)
    lr_monitor_callback = pl.callbacks.LearningRateMonitor()
    checkpoint_callback = ModelCheckpoint(dirpath=save_path, monitor="val_loss with TF", save_top_k=1)
    print('Starting Training')
    trainer = pl.Trainer(gpus=[6], num_nodes=1, precision=32,
                         logger=wandb_logger,
                         check_val_every_n_epoch=1,
                         #overfit_batches=5,
                         default_root_dir=save_path,
                         log_every_n_steps=20,
                         auto_lr_find=True,
                         callbacks=[lr_monitor_callback, checkpoint_callback],
                         max_epochs=50,
                         gradient_clip_val=5.,
                         #accelerator='ddp',
                         )                                
    trainer.tune(model, train_loader, val_loader)
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)

    '''
    i = 0
    output = []
    num_ims = 10
    model.to('cpu')
    for ((img_tensor, fac_cap, img_name), hum_cap, rom_cap) in val_data_visu:
        line = []
        if i == num_ims:
            break
        line.append(img_name)
        image = torch.unsqueeze(img_tensor, 0)
        img_feats = model.image_encoder(image.float())
        
        gt_caps = [fac_cap, hum_cap, rom_cap]
        gt_cap = torch.unsqueeze(fac_cap, 0)
        cap , _= model.captioner(img_feats, gt_cap.long(), 1.0)

        cap = torch.squeeze(cap)
        line.append(cap_to_text(cap, vocab))

        for gt_cap in gt_caps:
            line.append(cap_to_text_gt(gt_cap, vocab))
        
        output.append(line)
        i += 1


    model.build_html(output, save_html)
    '''