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
    one_shot_images = 'data/one_shot_images/'
    one_shot_captions = 'data/one_shot_captions.txt'
    hn_1hot = "/cortex/users/cohenza4/checkpoint/HN/one_hot/epoch=49-step=8149.ckpt"
    hn_emb = "/cortex/users/cohenza4/checkpoint/HN/embedding/epoch=16-step=2770.ckpt"
    hn_hist = "/cortex/users/cohenza4/checkpoint/HN/histograme/epoch=25-step=4237.ckpt"
    hn_hist_log = "/cortex/users/cohenza4/checkpoint/HN/histograme_log/epoch=32-step=5378.ckpt"
    hn_tfidf = "/cortex/users/cohenza4/checkpoint/HN/histograme_tfidf/epoch=22-step=3748.ckpt"
    hn_jsd = "/cortex/users/cohenza4/checkpoint/HN/JSD/epoch=19-step=3259.ckpt"
    gru_emb = "/cortex/users/cohenza4/checkpoint/GRU/emb/epoch=53-step=8801.ckpt"
    gru_no_emb = "/cortex/users/cohenza4/checkpoint/GRU/no_emb/epoch=49-step=8149.ckpt"
    # data
    with open("data/vocab_CC.pkl", 'rb') as f:
        vocab = pickle.load(f)
    print('Prepairing Data')
    test_data = get_dataset(one_shot_images, one_shot_captions, vocab)
    test_loader = DataLoader(test_data, batch_size=20, num_workers=2,
                            shuffle=False, collate_fn= collate_fn)

    list_domain = get_domain_list(cap_dir_train, cap_dir_val)   
    list_domain_zeroshot = get_domain_list(one_shot_captions, '')                
    # model
    #'histograme' 'histograme log' 'histograme tfidf' 'JSD', "embedding", "one hot"
    
    domain_emb = 'one hot'
    model = HyperNetCC(200, 200, 200, len(vocab), vocab, list_domain, 0.001, False, 0.3, 10, domain_emb, zero_shot=True, list_zeroshot=list_domain_zeroshot)  
    model = model.load_from_checkpoint(checkpoint_path=hn_1hot, vocab=vocab, list_domain=list_domain, embedding=domain_emb, zero_shot=True, list_zeroshot=list_domain_zeroshot)
    '''
    domain_emb = True                      
    model = Gru(200, 200, 200, len(vocab), vocab, list_domain, 0.001, domain_embed=domain_emb, zero_shot=True, list_zeroshot=list_domain_zeroshot)   
    model = model.load_from_checkpoint(checkpoint_path=gru_no_emb, vocab=vocab, list_domain=list_domain, domain_embed=domain_emb, zero_shot=True, list_zeroshot=list_domain_zeroshot)
    '''
    

    wandb_logger = WandbLogger(save_dir='/cortex/users/cohenza4')
    wandb_logger.log_hyperparams(model.hparams)
    print('Starting Test')
    trainer = pl.Trainer(gpus=[0],num_nodes=1, precision=32)                                
    trainer.test(model, test_loader)
  