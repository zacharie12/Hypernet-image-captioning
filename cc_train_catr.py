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
from models.encoder import EncoderCNN
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import copy
import numpy as np
from baseline_dataloader import collate_fn, get_dataset
import build_vocab
from build_vocab import Vocab
import pickle
import random
from pytorch_lightning.loggers import WandbLogger   
from pytorch_lightning.callbacks import ModelCheckpoint
from utils import CustomBertTokenizer, set_all_parameters, flip_parameters_to_tensors, WordVectorLoader, cap_to_text, cap_to_text_gt, cap_to_text_gt_batch, metric_score, metric_score_test, get_domain_list, get_hist_embedding, tfidf_hist, get_jsd_tsne
from pytorch_lightning.callbacks import ModelCheckpoint
import datasets
import dominate
from dominate.tags import *
from baseline import utils_baseline, caption
from baseline.configuration import Config



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Cart(pl.LightningModule):
    def __init__(self, vocab, lr=5e-5, path_bert='bert.pth'):
        super(Cart, self).__init__()
        self.vocab = vocab
        self.len_vocab = len(vocab)
        self.lr = lr
        #self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.captioner, self.criterion = caption.build_model(Config)
        
        self.metrics = ['bleu', 'meteor', 'rouge']
        self.metrics = [datasets.load_metric(name) for name in self.metrics]

 

    def forward(self, images, masked_token_ids, token_type_ids, position_ids, attention_mask, padding):
        image_features = self.image_encoder(images.float())
        pred_scores = self.generator(image_features, masked_token_ids, token_type_ids, position_ids, attention_mask, padding)
        pred_scores = pred_scores.contiguous().view(-1, self.len_vocab)
        return pred_scores

    def configure_optimizers(self):
        params = list(self.parameters())
        #params = list(self.generator.parameters())
        optimizer = torch.optim.Adam(params, lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, cooldown=2, factor=0.5)
        return [optimizer], [{'scheduler': scheduler, 'monitor': 'val_loss', 'interval': 'epoch'}]
        #return optimizer
  
    def training_step(self, train_batch, batch_idx):
        images, token_type_ids, input_token_ids, masked_token_ids, domain  = train_batch
        
        bleu1, bleu2, bleu3, bleu4, meteor, rouge = metric_score(gt_token_ids.unsqueeze(0), pred_scores.unsqueeze(0), self.vocab, self.metrics)
        self.log('meteor train', meteor)
        self.log('bleu 1 train', bleu1)
        self.log('bleu 2 train', bleu2)
        self.log('bleu 3 train', bleu3)
        self.log('bleu 4 train', bleu4)
        self.log('rouge train', rouge)

        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        images, token_type_ids, input_token_ids, masked_token_ids, domain  = val_batch
        seq_length = input_token_ids.size(1)
        batch_size = input_token_ids.size(0)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_token_ids)
        attention_mask = (masked_token_ids != self.PAD).float()
        mask_position = (masked_token_ids == self.MASK).to(torch.long).view(-1)
        mask_position = mask_position.nonzero().squeeze()

        padding = torch.zeros([len(images), 1, 1, 49]).to(device)
        pred_scores = self.forward(images, masked_token_ids, token_type_ids, position_ids, attention_mask, padding)
        #pred_scores = pred_scores.contiguous().view(-1, self.num_tokens)
        pred_scores = pred_scores[mask_position]
        gt_token_ids = input_token_ids.view(-1)[mask_position]
        loss = self.loss(pred_scores, gt_token_ids)
        bleu1, bleu2, bleu3, bleu4, meteor, rouge = metric_score(gt_token_ids.unsqueeze(0), pred_scores.unsqueeze(0), self.vocab, self.metrics)
        self.log('meteor val', meteor)
        self.log('bleu 1 val', bleu1)
        self.log('bleu 2 val', bleu2)
        self.log('bleu 3 val', bleu3)
        self.log('bleu 4 val', bleu4)
        self.log('rouge val', rouge)
        self.log('val_loss', loss)

    def test_step(self, test_batch, batch_idx):
        images, token_type_ids, input_token_ids, masked_token_ids, domain  = test_batch
        seq_length = input_token_ids.size(1)
        batch_size = input_token_ids.size(0)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_token_ids)
        attention_mask = (masked_token_ids != self.PAD).float()
        mask_position = (masked_token_ids == self.MASK).to(torch.long).view(-1)
        mask_position = mask_position.nonzero().squeeze()

        padding = torch.zeros([len(images), 1, 1, 49]).to(device)
        pred_scores = self.forward(images, masked_token_ids, token_type_ids, position_ids, attention_mask, padding)
        #pred_scores = pred_scores.contiguous().view(-1, self.num_tokens)
        pred_scores = pred_scores[mask_position]
        gt_token_ids = input_token_ids.view(-1)[mask_position]
        bleu1, bleu2, bleu3, bleu4, meteor, rouge = metric_score(gt_token_ids.unsqueeze(0), pred_scores.unsqueeze(0), self.vocab, self.metrics)
        self.log('meteor test', meteor)
        self.log('bleu 1 test', bleu1)
        self.log('bleu 2 test', bleu2)
        self.log('bleu 3 test', bleu3)
        self.log('bleu 4 test', bleu4)
        self.log('rouge test', rouge)


if __name__ == "__main__":
    glove_path = "/cortex/users/cohenza4/glove.6B.200d.txt"
    img_dir_train = 'data/200_conceptual_images_train/'
    img_dir_val_test = 'data/200_conceptual_images_val/'
    cap_dir_train = 'data/train_cap_100.txt'
    cap_dir_val = 'data/val_cap_100.txt'
    cap_dir_test = 'data/test_cap_100.txt'
    save_path = "/cortex/users/cohenza4/checkpoint/baseline/"
    # data
    with open("data/vocab.pkl", 'rb') as f:
        vocab = pickle.load(f)
    train_data = get_dataset(img_dir_train, cap_dir_train, vocab)
    val_data = get_dataset(img_dir_val_test, cap_dir_val, vocab)
    test_data = get_dataset(img_dir_val_test, cap_dir_test, vocab)

    train_loader = DataLoader(train_data, batch_size=32, num_workers=2,
                            shuffle=False, collate_fn= collate_fn)
    val_loader = DataLoader(val_data, batch_size=2, num_workers=2,
                            shuffle=False, collate_fn= collate_fn)
    test_loader = DataLoader(test_data, batch_size=2, num_workers=2,
                            shuffle=False, collate_fn= collate_fn)
    model = baseline(vocab)                       
    print(model)
    wandb_logger = WandbLogger(save_dir='/cortex/users/cohenza4')
    lr_monitor_callback = pl.callbacks.LearningRateMonitor()
    checkpoint_callback = ModelCheckpoint(dirpath=save_path, monitor="val_loss", save_top_k=1)
    print('Starting Training')
    trainer = pl.Trainer(gpus=[2], num_nodes=1, precision=32,
                         logger=wandb_logger,
                         #overfit_batches = 1,
                         check_val_every_n_epoch=1,
                         default_root_dir=save_path,
                         log_every_n_steps=20,
                         auto_lr_find=False,
                         callbacks=[lr_monitor_callback, checkpoint_callback],
                         #accelerator='ddp',
                         max_epochs=30,
                         gradient_clip_val=5.,
                         )
    trainer.tune(model, train_loader, val_loader)
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)
   