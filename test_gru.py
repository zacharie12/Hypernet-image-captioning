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
from train_attention_gru import CaptionAttentionGru
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


if __name__ == "__main__":
    img_path = "data/flickr7k_images"
    cap_path = "data/factual_train.txt"
    cap_path_humor = "data/humor/funny_train.txt"
    cap_path_romantic = "data/romantic/romantic_train.txt"
    glove_path = "/cortex/users/cohenza4/glove.6B.100d.txt"
    gru_factual = "/cortex/users/cohenza4/checkpoint_gru/factual/epoch=51-step=4488.ckpt"
    gru_humour = "/cortex/users/cohenza4/checkpoint_gru/gru_pretrain_humour/epoch=21-step=1848.ckpt"
    gru_romantic = "/cortex/users/cohenza4/checkpoint_gru/gru_pretrain_romantic/epoch=23-step=2024.ckpt"
    # data
    with open("data/vocab.pkl", 'rb') as f:
        vocab = pickle.load(f)
    print('Prepairing Data')
    orig_dataset = get_dataset(img_path, cap_path, vocab)
    humor_dataset = get_styled_dataset(cap_path_humor, vocab)
    romantic_dataset = get_styled_dataset(cap_path_romantic, vocab)

    data_concat = ConcatDataset(orig_dataset, humor_dataset, romantic_dataset)

    lengths = [int(len(data_concat)*0.8), int(len(data_concat)*0.1),
               len(data_concat) - (int(len(data_concat)*0.8) + int(len(data_concat)*0.1))]
    train_data, val_data, test_data = torch.utils.data.random_split(data_concat, lengths)

    test_loader = DataLoader(test_data, batch_size=1, num_workers=1,
                            shuffle=False, collate_fn=lambda x: flickr_collate_style(x, 'factual'))

    # model
    model = CaptionAttentionGru(200, 200, 200, len(vocab), vocab, lr=0.0003)
    model = model.load_from_checkpoint(checkpoint_path=gru_factual, vocab=vocab)
    
 
    wandb_logger = WandbLogger(save_dir='/cortex/users/cohenza4')
    wandb_logger.log_hyperparams(model.hparams)
    print('Starting Test')
    trainer = pl.Trainer(gpus=[1], num_nodes=1, precision=32)                                
    trainer.test(model, test_loader)
  