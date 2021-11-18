import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from torch.utils.data import DataLoader
from data_loader import get_dataset, get_styled_dataset, flickr_collate_fn, ConcatDataset, FlickrCombiner
from build_vocab import Vocab
import pickle
import numpy as np
from pytorch_lightning.loggers import WandbLogger
from models.decoderlstm import  AttentionGru
from train_attention_gru import CaptionAttentionGru
from utils import set_all_parameters, flip_parameters_to_tensors, WordVectorLoader, cap_to_text, cap_to_text_gt, metric_score
from hypernet_attention import HyperNet
from itertools import cycle
import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger   
from pytorch_lightning.callbacks import ModelCheckpoint


if __name__ == "__main__":
    #lstm_path = ["/cortex/users/algiser/caption-hn"]
    gru_path_factual = "/cortex/users/cohenza4/checkpoint_gru/small_factual/epoch=43-step=1892.ckpt"
    gru_path_romantic = "/cortex/users/cohenza4/checkpoint_gru/pretrain/romantic_tf/epoch=15-step=660.ckpt"
    gru_path_humor = "/cortex/users/cohenza4/checkpoint_gru/pretrain/humor_tf/epoch=11-step=484.ckpt"
    save_path1 = "/cortex/users/cohenza4/checkpoint_gru/hn/hn_all_style_tf.ckpt"
    save_path2 = "/cortex/users/cohenza4/checkpoint_gru/hn/hn_all_tf.pt"

    with open("data/vocab.pkl", 'rb') as f:
        vocab = pickle.load(f)

    styles = ['factual', 'romantic', 'humour']
    #styles = ['factual']
    #lstm = [DecoderRNN(100, 150, len(vocab), 2) for style in styles]
    #gru = DecoderGRU(200, 150, len(vocab), 2, dropout=False).to('cpu')
    rnn_fact = CaptionAttentionGru(200, 100, 180, len(vocab), vocab).to('cuda:6')
    rnn_rom = CaptionAttentionGru(200, 100, 180, len(vocab), vocab).to('cuda:6')
    rnn_humor = CaptionAttentionGru(200, 100, 180, len(vocab), vocab).to('cuda:6')

    #lstms = [model.load_state_dict(lstm_path[i])
    #         for i, model in enumerate(lstms)]
    #rnn.load_state_dict(torch.load(gru_path)) 
    print('Loading RNN')
    rnn_fact = rnn_fact.load_from_checkpoint(checkpoint_path=gru_path_factual, vocab=vocab).to('cuda:6')
    rnn_rom = rnn_rom.load_from_checkpoint(checkpoint_path=gru_path_romantic, vocab=vocab).to('cuda:6')
    rnn_humor = rnn_humor.load_from_checkpoint(checkpoint_path=gru_path_humor, vocab=vocab).to('cuda:6')
    rnns = [rnn_fact, rnn_rom, rnn_humor]
    #rnns = [rnn_fact]
    # model
    print('Initializing Hypernet')
    model = HyperNet(200, 100, 180, len(vocab), vocab).to('cuda:6')
    model.captioner.embed = rnn_fact.captioner.embed
    print('Finished initializing Hypernet')
    hn_heads = model.hn_heads
    hn_base = model.hn_base

    criterion = nn.MSELoss()
    params = list(hn_heads.parameters())
    params.extend(list(hn_base.parameters()))
    optimizer = torch.optim.SGD(params, lr=0.1)

    iteration = 0
    print('Starting training')
    flag = True
    for (style, rnn) in cycle(zip(styles, rnns)):

        optimizer.zero_grad()
        style = torch.tensor([vocab(style)]) 
        style = style.type(torch.LongTensor)
        style = style.to(model.device)
        style_embed = model.captioner.embed(style)

        base = hn_base(style_embed)
        i = 0
        loss = 0.
        for name, W in rnn.captioner.gru.named_parameters():
            if name == 'embed.weight':
                continue
            if name == 'fc_out.weight':
                continue
            if name == 'fc_out.bias':
                continue
            
            
            w_size = len(W.flatten())
            i_head = hn_heads[i](base)
            loss += criterion(i_head.flatten(), W.flatten()) 
            i += 1
        
        if iteration % 1000 == 0:
            print(f'iteration {iteration}: loss={loss.item()}')
        if loss.item() < 1e-4 and flag:
            print('Saving hypernet 1')
            torch.save(model.state_dict(), save_path1)
            print('hypernet 1 saved')
            flag = False
        elif  loss.item() < 1e-8:  
            print('Saving hypernet 2')
            torch.save(model.state_dict(), save_path2)
            print('hypernet 2 saved') 
            exit()
 
        iteration += 1
        loss.backward()
        optimizer.step()

        


    