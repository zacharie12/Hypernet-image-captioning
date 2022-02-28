import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils import data
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torchvision
from torchvision import transforms
from models.decoderlstm import AttentionGru
from train_attention_gru import CaptionAttentionGru
from models.encoder import EncoderCNN
from bert_text_classifier import BertClassifer
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import copy
from data_loader_combine import get_dataset, ConcatDataset, combine_collate_fn, collate_fn_test
import build_vocab
from build_vocab import Vocab
import pickle
import random
from pytorch_lightning.loggers import WandbLogger   
from utils import set_all_parameters, flip_parameters_to_tensors, WordVectorLoader, cap_to_text, cap_to_text_gt, metric_score, metric_score_test, get_domain_list, get_hist_embedding, tfidf_hist, get_jsd_tsne
from pytorch_lightning.callbacks import ModelCheckpoint
import datasets
import numpy as np
import random 
from train_hyper_combine import HyperNetCC
from train_gru_combine import Gru
import tldextract
from PIL import Image
import skimage.transform
import requests
import PIL
import cv2
from matplotlib import cm
from collections import Counter


if __name__ == "__main__":
    glove_path = "/cortex/users/cohenza4/glove.6B.200d.txt"
    img_dir_fliker = 'data/flickr7k_images/'
    img_dir_CC_val_test = 'data/200_conceptual_images_val/'
    caption_CC_train = 'data/train_cap_100.txt'
    caption_CC_test = 'data/test_cap_100.txt'
    caption_fac_test = 'data/fac_cap_test.txt'
    caption_hum_test = 'data/humor_cap_test.txt'
    caption_rom_test = 'data/rom_cap_test.txt'
    zero_shot_captions = 'data/one_shot_captions.txt'
    zero_shot_images = 'data/one_shot_images/'
    gru_no_emb = "/cortex/users/cohenza4/checkpoint/GRU_combine/emb/epoch=50-step=8312.ckpt"
    gru_emb = "/cortex/users/cohenza4/checkpoint/GRU_combine/no_emb/epoch=28-step=4726.ckpt"
    hn_hist = "/cortex/users/cohenza4/checkpoint/HN_combine/histograme/epoch=52-step=8638.ckpt"
    hn_hist_log = "/cortex/users/cohenza4/checkpoint/HN_combine/histograme_log/epoch=41-step=6845.ckpt"
    hn_hist_tfidf = "/cortex/users/cohenza4/checkpoint/HN_combine/histograme_tfidf/epoch=47-step=7823.ckpt"
    hn_jsd = "/cortex/users/cohenza4/checkpoint/HN_combine/JSD/epoch=51-step=8475.ckpt"
    hn_emb = "/cortex/users/cohenza4/checkpoint/HN_combine/embedding/epoch=34-step=5704.ckpt"
    hn_1hot = "/cortex/users/cohenza4/checkpoint/HN_combine/one_hot/epoch=29-step=4889.ckpt"

    with open("data/vocab.pkl", 'rb') as f:
        vocab = pickle.load(f)

    
    test_data_fac = get_dataset(img_dir_fliker, caption_fac_test, vocab, "Fliker factual")
    test_data_hum = get_dataset(img_dir_fliker, caption_hum_test, vocab, "Fliker style")
    test_data_rom = get_dataset(img_dir_fliker, caption_rom_test, vocab, "Fliker style")
    test_data_CC = get_dataset(img_dir_CC_val_test, caption_CC_test, vocab)
    test_data_CC_zeroshot = get_dataset(zero_shot_images, zero_shot_captions, vocab)

    test_data_concat = ConcatDataset(test_data_fac, test_data_hum, test_data_rom, test_data_CC, test_data_CC_zeroshot)

    test_loader_fac = DataLoader(test_data_concat, batch_size=32, num_workers=2, shuffle=False,  collate_fn=lambda x: collate_fn_test(x, 'factual'))
    test_loader_hum = DataLoader(test_data_concat, batch_size=32, num_workers=2, shuffle=False,  collate_fn=lambda x: collate_fn_test(x, 'humour'))
    test_loader_rom = DataLoader(test_data_concat, batch_size=32, num_workers=2, shuffle=False,  collate_fn=lambda x: collate_fn_test(x, 'romantic'))
    test_loader_CC = DataLoader(test_data_concat, batch_size=10, num_workers=2, shuffle=False,  collate_fn=lambda x: collate_fn_test(x, 'CC'))
    test_loader_CC_zeroshot = DataLoader(test_data_concat, batch_size=20, num_workers=2, shuffle=False,  collate_fn=lambda x: collate_fn_test(x, 'CC'))

    list_domain_cc = get_domain_list(caption_CC_train, zero_shot_captions)
    list_fliker = ['f', 'r', 'h']
    list_domain = list_domain_cc + list_fliker

    
    #'histograme' 'histograme log' 'histograme tfidf' 'JSD', "embedding", "one hot"
    domain_emb = 'JSD'
    model = HyperNetCC(200, 200, 200, len(vocab), vocab, list_domain, 0.001, False, 0.3, 10, domain_emb)
    model = model.load_from_checkpoint(checkpoint_path=hn_jsd, vocab=vocab, list_domain=list_domain, embedding=domain_emb)

    """
    domain_emb = True                      
    # model
    model = Gru(200, 200, 200, len(vocab), vocab, list_domain, 0.001, domain_embed=domain_emb)   
    model = model.load_from_checkpoint(checkpoint_path=gru_emb, vocab=vocab, list_domain=list_domain, domain_embed=domain_emb)
    
    """
    trainer = pl.Trainer(gpus=[0],num_nodes=1, precision=32)                                
    trainer.test(model, test_loader_fac)
    trainer.test(model, test_loader_hum)
    trainer.test(model, test_loader_rom)
    trainer.test(model, test_loader_CC)
    trainer.test(model, test_loader_CC_zeroshot)