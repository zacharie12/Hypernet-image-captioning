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
from cc_dataloader import ConceptualCaptions, collate_fn, get_dataset
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
from hypernet_attention import HyperNet
import tldextract
from PIL import Image
import skimage.transform
import requests
import PIL
import cv2
from matplotlib import cm
from cc_train_hypernet import HyperNetCC
from argparse import ArgumentParser


if __name__ == "__main__":

    parser = ArgumentParser()
    # paths
    '''
    parser.add_argument('--img_dir_train', type=str, default='data/200_conceptual_images_train/')
    parser.add_argument('--img_dir_val_test', type=str, default="data/200_conceptual_images_val/")
    parser.add_argument('--cap_dir_train', type=str, default='data/train_cap_100.txt')
    parser.add_argument('--cap_dir_val', type=str, default="data/val_cap_100.txt")
    parser.add_argument('--cap_dir_test', type=str, default="data/test_cap_100.txt")
    parser.add_argument('--domain_classifier_dir', type=str, default='/cortex/users/cohenza4/checkpoint/domain_classifier/')
    parser.add_argument('--glove_path', type=str, default="/cortex/users/cohenza4/glove.6B.200d.txt")
    parser.add_argument('--vocab_path', type=str, default="data/vocab.pkl")
    parser.add_argument('--save_path', type=str, default='/cortex/users/cohenza4/checkpoint/HN/debug/')
    parser.add_argument('--batch_size', type=int, default=32)
    '''
    parser.add_argument('--domain_emb', type=str, default='embedding')
    parser.add_argument('--mixup', type=bool, default=False)
    parser.add_argument('--alpha', type=int, default=0.3)
    parser.add_argument('--hyper_emb', type=int, default=10)
    parser.add_argument('--n_tsne', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.001)

    #parser = HyperNetCC.add_model_specific_args(parser)
    # trainer settings
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    '''
    img_dir_train = args.img_dir_train
    img_dir_val_test = args.img_dir_val_test
    cap_dir_train = args.cap_dir_train
    cap_dir_val = args.cap_dir_val
    cap_dir_test = args.cap_dir_test
    glove_path = args.glove_path
    save_path = args.save_path
    domain_classifier_dir = args.domain_classifier_dir
    '''
    glove_path = "/cortex/users/cohenza4/glove.6B.200d.txt"
    img_dir_train = 'data/200_conceptual_images_train/'
    img_dir_val_test = 'data/200_conceptual_images_val/'
    cap_dir_train = 'data/train_cap_100.txt'
    cap_dir_val = 'data/val_cap_100.txt'
    cap_dir_test = 'data/test_cap_100.txt'
    save_path = "/cortex/users/cohenza4/checkpoint/HN/debug/"
    domain_emb = args.domain_emb
    mixup = args.mixup
    alpha = args.alpha
    hyper_emb = args.hyper_emb
    n_tsne = args.n_tsne
    lr = args.lr
    
    # data
    with open("data/vocab.pkl", 'rb') as f:
        vocab = pickle.load(f)

    print('Prepairing Data')
    train_data = get_dataset(img_dir_train, cap_dir_train, vocab)
    val_data = get_dataset(img_dir_val_test, cap_dir_val, vocab)
    test_data = get_dataset(img_dir_val_test, cap_dir_test, vocab)
    #val_test_data = get_dataset(img_dir_val_test, cap_dir_val_test, vocab)
    #lengths = [int(len(val_test_data)*0.3), len(val_test_data) - (int(len(val_test_data)*0.3))]
    #print("lengths = ", lengths)
    #val_data, test_data = torch.utils.data.random_split(val_test_data, lengths)
    train_loader = DataLoader(train_data, batch_size=32, num_workers=2,
                            shuffle=False, collate_fn= collate_fn)
    val_loader = DataLoader(val_data, batch_size=8, num_workers=2,
                            shuffle=False, collate_fn= collate_fn)
    test_loader = DataLoader(test_data, batch_size=1, num_workers=2,
                            shuffle=False, collate_fn= collate_fn)
    list_domain = get_domain_list(cap_dir_train, cap_dir_val)
    model = HyperNetCC(200, 200, 200, len(vocab), vocab, list_domain, lr, mixup, alpha, hyper_emb, domain_emb, n_tsne)
    print('Loading GloVe Embedding')
    model.load_glove_emb(glove_path)

    wandb_logger = WandbLogger(save_dir='/cortex/users/cohenza4')
    wandb_logger.log_hyperparams(model.hparams)

    lr_monitor_callback = pl.callbacks.LearningRateMonitor()
    checkpoint_callback = ModelCheckpoint(dirpath='/cortex/users/cohenza4/checkpoint/HN/debug/', monitor="val_loss with TF")
    print('Starting Training')
    trainer = pl.Trainer.from_argparse_args(args, callbacks=[lr_monitor_callback, checkpoint_callback], logger=wandb_logger)
    trainer.tune(model, train_loader, val_loader)
    trainer.fit(model, train_loader, val_loader)
    #trainer.test(model, test_loader)
