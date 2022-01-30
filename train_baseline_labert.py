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
from Labert import Generator, LabelSmoothingLoss
from transformers.modeling_bert import BertConfig
#from utils import EOS, MASK, PAD, num_tokens, tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class baseline(pl.LightningModule):
    def __init__(self, vocab, lr=5e-5, path_bert='bert.pth'):
        super(baseline, self).__init__()

        self.vocab = vocab
        self.len_vocab = len(vocab)
        self.lr = lr
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.image_encoder = EncoderCNN()
        bert_config = BertConfig(type_vocab_size=len(config.boundaries) + 2)
        self.generator = Generator(bert_config)
        balance_weight =torch.ones(self.len_vocab, dtype=torch.float32)
        self.loss = LabelSmoothingLoss(len(vocab), balance_weight, 0.1)

 

    def forward(self, image, input_token_ids, attention_mask):
        image_features = self.image_encoder(image.float())
        pred_scores = generator(image_features, input_token_ids, attention_mask)
        pred_scores = pred_scores.contiguous().view(-1, self.len_vocab)
        return pred_scores

    def configure_optimizers(self):
        #params = list(self.linear.parameters())
        params = list(self.generator.parameters())
        optimizer = torch.optim.Adam(params, lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, cooldown=2, factor=0.5)
        return [optimizer], [{'scheduler': scheduler, 'monitor': 'val_loss', 'interval': 'epoch'}]
        #return optimizer
  
    def training_step(self, train_batch, batch_idx):
        imgs, caps, lengths, domains = train_batch
        for i in range(len(caps)):
            gt_idx = torch.squeeze(caps[i])
            text = cap_to_text_gt(gt_idx, self.vocab, tokenized=False)
            caption = self.tokenizer(text, 
                                padding='max_length', max_length = 30, truncation=True,
                                    return_tensors="pt") 
            mask = caption['attention_mask'].to(device)
            input_id = caption['input_ids'].squeeze(1).to(device)

            pred = self.forward(imgs, input_id, mask)
            loss += self.loss (pred, input_id)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        imgs, caps, lengths, domains = val_batch
        for i in range(len(caps)):
            gt_idx = torch.squeeze(caps[i])
            text = cap_to_text_gt(gt_idx, self.vocab, tokenized=False)
            caption = self.tokenizer(text, 
                                padding='max_length', max_length = 30, truncation=True,
                                    return_tensors="pt") 
            mask = caption['attention_mask'].to(device)
            input_id = caption['input_ids'].squeeze(1).to(device)

            pred = self.forward(imgs, input_id, mask)
            loss += self.loss (pred, input_id)
        self.log('val_loss', loss)

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
    val_loader = DataLoader(val_data, batch_size=8, num_workers=2,
                            shuffle=False, collate_fn= collate_fn)
    test_loader = DataLoader(test_data, batch_size=1, num_workers=2,
                            shuffle=False, collate_fn= collate_fn)
    model = baseline(vocab)                       
    print(model)
    wandb_logger = WandbLogger(save_dir='/cortex/users/cohenza4')
    lr_monitor_callback = pl.callbacks.LearningRateMonitor()
    checkpoint_callback = ModelCheckpoint(dirpath=save_path, monitor="val_loss", save_top_k=1)
    print('Starting Training')
    trainer = pl.Trainer(gpus=[1], num_nodes=1, precision=32,
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
    #trainer.test(model, test_loader)
   