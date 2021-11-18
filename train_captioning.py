import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from torch.utils.data import DataLoader
from data_loader import get_dataset, get_styled_dataset, flickr_collate_fn, ConcatDataset, FlickrCombiner
from build_vocab import Vocab
import pickle
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger   
from utils import set_all_parameters, flip_parameters_to_tensors, WordVectorLoader, cap_to_text, cap_to_text_gt, metric_score
from hypernet import HyperNet
from argparse import ArgumentParser

if __name__ == "__main__":

    parser = ArgumentParser()
    # paths
    parser.add_argument('--img_path', type=str, default="data/flickr7k_images")
    parser.add_argument('--cap_path', type=str, default="data/factual_train.txt")
    parser.add_argument('--cap_path_humor', type=str, default="data/humor/funny_train.txt")
    parser.add_argument('--cap_path_romantic', type=str, default="data/romantic/romantic_train.txt")
    parser.add_argument('--glove_path', type=str, default="/cortex/users/algiser/glove.6B.200d.txt")
    parser.add_argument('--vocab_path', type=str, default="data/vocab.pkl")
    parser.add_argument('--save_dir', type=str, default='/cortex/users/algiser')
    
    # model hparams
    parser = HyperNet.add_model_specific_args(parser)
    # trainer settings
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    img_path = args.img_path
    cap_path = args.cap_path
    cap_path_humor = args.cap_path_humor
    cap_path_romantic = args.cap_path_romantic
    glove_path = args.glove_path
    save_dir = args.save_dir
    # data
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    print('Prepairing Data')
    orig_dataset = get_dataset(img_path, cap_path, vocab)
    humor_dataset = get_styled_dataset(cap_path_humor, vocab)
    romantic_dataset = get_styled_dataset(cap_path_romantic, vocab)

    data_concat = ConcatDataset(orig_dataset, humor_dataset, romantic_dataset)
    lengths = [int(len(data_concat)*0.8),
               len(data_concat) - int(len(data_concat)*0.8)]
    train_data, val_data = torch.utils.data.random_split(data_concat, lengths)

    train_loader = DataLoader(train_data, batch_size=42, num_workers=12,
                              shuffle=False, collate_fn=flickr_collate_fn)
    val_loader = DataLoader(val_data, batch_size=42, num_workers=12,
                            shuffle=False, collate_fn=flickr_collate_fn)
    # model


    model = HyperNet(args.hidden_size, args.embed_size, len(vocab), vocab, args.num_layers, type=args.type)
    print('Loading GloVe Embedding')
    model.load_glove_emb(glove_path)

    wandb_logger = WandbLogger(save_dir=args.save_dir)
    wandb_logger.log_hyperparams(model.hparams)

    lr_monitor_callback = pl.callbacks.LearningRateMonitor()
    print('Starting Training')
    trainer = pl.Trainer.from_argparse_args(args, callbacks=[lr_monitor_callback], logger=wandb_logger)

    #trainer.tune(model, train_loader, val_loader)
    trainer.fit(model, train_loader, val_loader)
