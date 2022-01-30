from charset_normalizer import from_path
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
from train_attention_gru import CaptionAttentionGru
from hypernet_attention import HyperNet
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
from cc_train_hypernet import HyperNetCC
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
    hn_1hot = "/cortex/users/cohenza4/checkpoint/HN/one_hot/epoch=19-step=3259.ckpt"
    hn_emb = "/cortex/users/cohenza4/checkpoint/HN/embedding/epoch=47-step=7823.ckpt"
    #hn_hist = "/cortex/users/cohenza4/checkpoint/HN/emb/epoch=44-step=7334.ckpt"
    hn_hist_log = "/cortex/users/cohenza4/checkpoint/HN/histograme_log/epoch=24-step=4074.ckpt"
    hn_tfidf = "/cortex/users/cohenza4/checkpoint/HN/histograme_tfidf/epoch=21-step=3585.ckpt"
    hn_jsd = "/cortex/users/cohenza4/checkpoint/HN/JSD/epoch=25-step=4237.ckpt"
    # data
    with open("data/vocab.pkl", 'rb') as f:
        vocab = pickle.load(f)
    print('Prepairing Data')
    test_data = get_dataset(img_dir_val_test, cap_dir_test, vocab)


    test_loader = DataLoader(test_data, batch_size=1, num_workers=2,
                            shuffle=False, collate_fn= collate_fn)

    list_domain = get_domain_list(cap_dir_train, cap_dir_val)                 
    # model
    #'histograme' 'histograme log' 'histograme tfidf' 'JSD', "embedding", "one hot"
    domain_emb = 'JSD'
    model = HyperNetCC(200, 200, 200, len(vocab), vocab, list_domain, 0.001, False, 0.3, 10, domain_emb)  
    model = model.load_from_checkpoint(checkpoint_path=hn_jsd, vocab=vocab, list_domain=list_domain, embedding=domain_emb)
    '''
    rnn = CaptionAttentionGru(200, 200, 200, len(vocab), vocab)
    rnn = rnn.load_from_checkpoint(checkpoint_path=gru_path, vocab=vocab)
    model.image_encoder = rnn.image_encoder
    model.captioner.feature_fc = rnn.captioner.feature_fc
    model.captioner.embed = rnn.captioner.embed
    model.captioner.fc = rnn.captioner.fc
    model.captioner.attention = rnn.captioner.attention
    model.captioner.init_h = rnn.captioner.init_h
    '''

    

    wandb_logger = WandbLogger(save_dir='/cortex/users/cohenza4')
    wandb_logger.log_hyperparams(model.hparams)
    print('Starting Test')
    trainer = pl.Trainer(gpus=[4],num_nodes=1, precision=32)                                
    trainer.test(model, test_loader)
  