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
from data_loader import get_dataset, get_styled_dataset, flickr_collate_fn, ConcatDataset, FlickrCombiner, flickr_collate_style
import build_vocab
from build_vocab import Vocab
import pickle
import random
from pytorch_lightning.loggers import WandbLogger   
from pytorch_lightning.callbacks import ModelCheckpoint
from utils import set_all_parameters, flip_parameters_to_tensors, WordVectorLoader, cap_to_text, cap_to_text_gt, metric_score, metric_score_test, get_domain_list
import datasets
import dominate
from dominate.tags import *
from cc_train_gru import Gru
from cc_dataloader import ConceptualCaptions, collate_fn, get_dataset


if __name__ == "__main__":
    glove_path = "/cortex/users/cohenza4/glove.6B.200d.txt"
    img_dir_train = 'data/200_conceptual_images_train/'
    img_dir_val_test = 'data/200_conceptual_images_val/'
    cap_dir_train = 'data/train_cap_100.txt'
    cap_dir_val = 'data/val_cap_100.txt'
    cap_dir_test = 'data/test_cap_100.txt'
    gru_emb = "/cortex/users/cohenza4/checkpoint/GRU/emb/epoch=44-step=7334.ckpt"
    gru_no_emb = "/cortex/users/cohenza4/checkpoint/GRU/no_emb/epoch=39-step=6519.ckpt"
    # data
    with open("data/vocab.pkl", 'rb') as f:
        vocab = pickle.load(f)
    print('Prepairing Data')
    test_data = get_dataset(img_dir_val_test, cap_dir_test, vocab)


    test_loader = DataLoader(test_data, batch_size=1, num_workers=2,
                            shuffle=False, collate_fn= collate_fn)

    list_domain = get_domain_list(cap_dir_train, cap_dir_val)  
    domain_emb = False                      
    # model
    model = Gru(200, 200, 200, len(vocab), vocab, list_domain, 0.001, domain_embed=domain_emb)   
    model = model.load_from_checkpoint(checkpoint_path=gru_no_emb, vocab=vocab, list_domain=list_domain, domain_embed=domain_emb)
    
 
    wandb_logger = WandbLogger(save_dir='/cortex/users/cohenza4')
    wandb_logger.log_hyperparams(model.hparams)
    print('Starting Test')
    trainer = pl.Trainer(gpus=[1], num_nodes=1, precision=32)                                
    trainer.test(model, test_loader)
  