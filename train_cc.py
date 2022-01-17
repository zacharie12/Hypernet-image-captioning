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
from conceptual_dataloader import ConceptualCaptions, pad_sequence, collate_fn_cc, Rescale
import build_vocab
from build_vocab import Vocab
import pickle
import random
from pytorch_lightning.loggers import WandbLogger   
from utils import set_all_parameters, flip_parameters_to_tensors, WordVectorLoader, cap_to_text, cap_to_text_gt, metric_score, metric_score_test
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class HyperNetCC(pl.LightningModule):
    def __init__(self, feature_size, embed_size, hidden_size, vocab_size, vocab, num_layers=1, lr=1e-6, mixup=False, alpha=0.3, hyper_emb=10):
        super().__init__()

        self.hparams['feature_size'] = feature_size
        self.hparams['vocab_size'] = vocab_size
        self.hparams['embed_size'] = embed_size
        self.hparams['hidden_size'] = hidden_size
        self.vocab = vocab
        self.hparams['lr'] = lr
        self.hparams['num_layers'] = num_layers
        self.teacher_forcing_proba = 0.0
        self.beam_size = 3
        self.metrics = ['bleu', 'meteor', 'rouge']
        self.mixup = mixup
        self.metrics = [datasets.load_metric(name) for name in self.metrics]
        self.alpha = alpha
        self.hypernet = HyperNet(feature_size, feature_size, feature_size, vocab_size, vocab, num_layers=1, lr=1e-6, mixup=False, alpha=0.3, cc=True, hyper_emb=hyper_emb)
        self.transforme = transforms.Compose([
            Rescale((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        self.encodeur = {'gettyimages':[0], 'shutterstock':[1], 'dailymail':[2], 'pinimg':[3],'123rf':[4], 'wordpress':[5], 'alamy':[6], 'picdn':[7], 'istockphoto':[8]}
        self.embed = nn.Embedding(len(self.encodeur), hyper_emb)

    def configure_optimizers(self):
        params = list(self.hypernet.hn_heads.parameters())
        params.extend(list(self.hypernet.hn_base.parameters()))
        params.extend(list(self.hypernet.captioner.feature_fc.parameters()))
        params.extend(list(self.hypernet.captioner.embed.parameters()))
        params.extend(list(self.hypernet.captioner.fc.parameters()))
        params.extend(list(self.hypernet.captioner.attention.parameters()))
        params.extend(list(self.hypernet.captioner.init_h.parameters()))
        optimizer = torch.optim.Adam(params, lr=self.hparams['lr'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, cooldown=2, factor=0.5)
        return [optimizer], [{'scheduler': scheduler, 'monitor': 'val_loss with TF', 'interval': 'epoch'}]

    def load_glove_emb(self, emb_path):
        loader = WordVectorLoader(self.hparams['embed_size'])
        loader.load_glove(emb_path)
        emb_mat = loader.generate_embedding_matrix(self.vocab.w2i,
                                                   self.vocab.ix-1, 'norm')
        emb_mat = torch.FloatTensor(emb_mat)
        self.hypernet.captioner.embed = self.hypernet.captioner.embed.from_pretrained(emb_mat,
                                                                    freeze=False)

    def training_step(self, train_batch, batch_idx):
        url, caps, lengths, domain = train_batch
        caps_pred_list = []
        loss = torch.tensor(0.0).to(device)
        for i in range(len(domain)):
            try:
                size_urls = len(url[i])
                search_url = url[i][:size_urls - 1]
                im = Image.open(requests.get(search_url, stream=True).raw)
                style= torch.tensor(self.encodeur[domain[i]])
                style = style.type(torch.LongTensor)
                cap =  torch.unsqueeze(caps[i], dim=0)
            except PIL.UnidentifiedImageError:
                search_url = 'https://ak6.picdn.net/shutterstock/videos/318736/thumb/1.jpg'
                im = Image.open(requests.get(search_url, stream=True).raw)
                cap = torch.tensor([[1.0000e+00, 6.3300e+02, 3.0000e+00, 1.1370e+03, 4.6000e+01, 2.6000e+01,
                                    6.2500e+02, 2.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00]]).to(device)
                style = torch.LongTensor([7])
            im = np.array(im)
            #if image grayscale convert to RGB
            if len(im.shape) == 2:
                colmap = cm.get_cmap('viridis', 256)
                np.savetxt('cmap.csv', (colmap.colors[...,0:3]*255).astype(np.uint8), fmt='%d', delimiter=',')
                lut = np.loadtxt('cmap.csv', dtype=np.uint8, delimiter=',')
                result = np.zeros((*im.shape,3), dtype=np.uint8)
                np.take(lut, im, axis=0, out=result)
                im = Image.fromarray(result)
                im = np.array(im)
            style = style.to(self.device)
            style_embed = self.embed(style)
            captioner = self.hypernet.forward(style_embed)
            #if grayscale image change to RGB

            image = self.transforme(im).to(device)
            img_feats = self.hypernet.image_encoder(torch.unsqueeze(image.float(), dim=0))
            caps_pred, _ = self.hypernet.captioner(img_feats, cap.long(), self.teacher_forcing_proba)
            caps_pred_list.append(caps_pred)
            loss +=  F.cross_entropy(caps_pred.view(-1, self.hparams['vocab_size']), cap.view(-1).long(), ignore_index=self.vocab.w2i['<pad>'])

        bleu1, bleu2, bleu3, bleu4, meteor, rouge = metric_score(caps, caps_pred_list, self.vocab, self.metrics)
        self.log('meteor train', meteor)
        self.log('bleu 1 train', bleu1)
        self.log('bleu 2 train', bleu2)
        self.log('bleu 3 train', bleu3)
        self.log('bleu 4 train', bleu4)
        self.log('rouge train', rouge)
        self.log('train_loss', loss)
        self.log('LR', self.hparams['lr'])
        return loss


    def validation_step(self, val_batch, batch_idx):
        url, caps, lengths, domain = val_batch
        caps_pred_list, caps_pre_tf_list = [], []
        loss = torch.tensor(0.0).to(device)
        for i in range(len(domain)):
            try:
                size_urls = len(url[i])
                search_url = url[i][:size_urls - 1]
                im = Image.open(requests.get(search_url, stream=True).raw)
                style= torch.tensor(self.encodeur[domain[i]])
                style = style.type(torch.LongTensor)
                cap =  torch.unsqueeze(caps[i], dim=0)
            except PIL.UnidentifiedImageError:
                search_url = 'https://ak6.picdn.net/shutterstock/videos/318736/thumb/1.jpg'
                im = Image.open(requests.get(search_url, stream=True).raw)
                cap = torch.tensor([[1.0000e+00, 6.3300e+02, 3.0000e+00, 1.1370e+03, 4.6000e+01, 2.6000e+01,
                                    6.2500e+02, 2.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00]]).to(device)
                style = torch.LongTensor([7])
            im = np.array(im)
            if len(im.shape) == 2:
                colmap = cm.get_cmap('viridis', 256)
                np.savetxt('cmap.csv', (colmap.colors[...,0:3]*255).astype(np.uint8), fmt='%d', delimiter=',')
                lut = np.loadtxt('cmap.csv', dtype=np.uint8, delimiter=',')
                result = np.zeros((*im.shape,3), dtype=np.uint8)
                np.take(lut, im, axis=0, out=result)
                im = Image.fromarray(result)
                im = np.array(im)
            style = style.to(self.device)
            style_embed = self.embed(style)
            captioner = self.hypernet.forward(style_embed)
            #if grayscale image change to RGB
            image = self.transforme(im).to(device)
            img_feats = self.hypernet.image_encoder(torch.unsqueeze(image.float(), dim=0))
            caps_pred_teacher_forcing, _ = captioner(img_feats, cap.long(), self.teacher_forcing_proba)
            caps_pre_tf_list.append(caps_pred_teacher_forcing)
            caps_pred, _ = captioner(img_feats, cap.long(), 1.0)
            caps_pred_list.append(caps_pred)

            # caps = nn.functional.one_hot(caps.to(torch.int64), num_classes=self.voc_size).float()
            loss += F.cross_entropy(caps_pred.view(-1, self.hparams['vocab_size']),
                                cap.view(-1).long(), ignore_index=self.vocab.w2i['<pad>'])

            loss_tf = F.cross_entropy(caps_pred_teacher_forcing.view(-1, self.hparams['vocab_size']),
                               cap.view(-1).long(), ignore_index=self.vocab.w2i['<pad>'])
        
        bleu1, bleu2, bleu3, bleu4, meteor, rouge = metric_score(caps, caps_pred_list, self.vocab, self.metrics)
        
        self.log('val_loss', loss, sync_dist=True)
        self.log('meteor', meteor)
        self.log('bleu 1', bleu1)
        self.log('bleu 2', bleu2)
        self.log('bleu 3', bleu3)
        self.log('bleu 4', bleu4)
        self.log('rouge', rouge)


        
        self.log('val_loss with TF', loss_tf, sync_dist=True)

    def test_step(self, test_batch, batch_idx):
        url, caps, lengths, domain = test_batch
        caps_pred_list, caps_pred_list_beam = [], []
        for i in range(len(domain)):
            try:
                size_urls = len(url[i])
                search_url = url[i][:size_urls - 1]
                im = Image.open(requests.get(search_url, stream=True).raw)
                style= torch.tensor(self.encodeur[domain[i]])
                style = style.type(torch.LongTensor)
                cap =  torch.unsqueeze(caps[i], dim=0)
            except PIL.UnidentifiedImageError:
                search_url = 'https://ak6.picdn.net/shutterstock/videos/318736/thumb/1.jpg'
                im = Image.open(requests.get(search_url, stream=True).raw)
                cap = torch.tensor([[1.0000e+00, 6.3300e+02, 3.0000e+00, 1.1370e+03, 4.6000e+01, 2.6000e+01,
                                    6.2500e+02, 2.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00]]).to(device)
                style = torch.LongTensor([7])
            im = np.array(im)
            if len(im.shape) == 2:
                colmap = cm.get_cmap('viridis', 256)
                np.savetxt('cmap.csv', (colmap.colors[...,0:3]*255).astype(np.uint8), fmt='%d', delimiter=',')
                lut = np.loadtxt('cmap.csv', dtype=np.uint8, delimiter=',')
                result = np.zeros((*im.shape,3), dtype=np.uint8)
                np.take(lut, im, axis=0, out=result)
                im = Image.fromarray(result)
                im = np.array(im)
            style = style.to(self.device)
            style_embed = self.embed(style)
            self.hypernet.captioner = self.hypernet.forward(style_embed)
            image = self.transforme(im)
            features = self.hypernet.image_encoder(image.float())
            encoder_out = self.hypernet.captioner.feature_fc(features)
            encoder_dim = encoder_out.size(-1)
            encoder_out = encoder_out.view(1, -1, encoder_dim) 
            num_pixels = encoder_out.size(1)
            k = self.beam_size
            vocab_size = self.hparams['vocab_size']
            encoder_out = encoder_out.expand(k, num_pixels, encoder_dim) 
            k_prev_words = torch.LongTensor([[0]] * k).to(device)
            seqs = k_prev_words
            top_k_scores = torch.zeros(k, 1).to(device)
            complete_seqs = []
            complete_seqs_scores = []
            step = 1
            h = self.hypernet.captioner.init_hidden(encoder_out)
            while True:
                embeddings = self.hypernet.captioner.embed(k_prev_words).squeeze(1)  
                if k_prev_words[0][0] == 0:
                    embeddings[:][:][:][:][:][:] = 0
                context, atten_weight = self.hypernet.captioner.attention(encoder_out, h) 
                input_concat = torch.cat([embeddings, context], 1)
                h = self.hypernet.captioner.gru(input_concat, h)  
                scores = self.hypernet.captioner.fc(h)  
                scores = F.log_softmax(scores, dim=1)
                # top_k_scores: [s, 1]
                scores = top_k_scores.expand_as(scores) + scores  # [s, vocab_size]
                # For the first step, all k points will have the same scores (since same k previous words, h, c)
                if step == 1:
                    top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
                else:
                    # Unroll and find top scores, and their unrolled indices
                    top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

                # Convert unrolled indices to actual indices of scores
                prev_word_inds = top_k_words // vocab_size  # (s)
                next_word_inds = top_k_words % vocab_size  # (s)
                # Add new words to sequences
                seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
                # Which sequences are incomplete (didn't reach <end>)?
                incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                                    next_word != self.vocab.w2i['</s>']]
                complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))
                # Set aside complete sequences
                if len(complete_inds) > 0:
                    Caption_End = True
                    compute = True
                    complete_seqs.extend(seqs[complete_inds].tolist())
                    complete_seqs_scores.extend(top_k_scores[complete_inds])
                k -= len(complete_inds)  # reduce beam length accordingly
                # Proceed with incomplete sequences
                if k == 0:
                    break

                seqs = seqs[incomplete_inds]
                h = h[prev_word_inds[incomplete_inds]]
                encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
                top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
                k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)
                # Break if things have been going on too long
                if step > 50:
                    Caption_End = True
                    compute = False
                    break
                step += 1

            # choose the caption which has the best_score.
            Caption_End = True
            assert Caption_End
            if compute:
                indices = complete_seqs_scores.index(max(complete_seqs_scores))
                caps_pred_beam = complete_seqs[indices]
                caps_pred_list_beam.append(caps_pred_beam)




            bleu1_beam, bleu2_beam, bleu3_beam, bleu4_beam, meteor_beam, rouge_beam = metric_score_test(caps, torch.tensor(caps_pred_list_beam), self.vocab, self.metrics)
            self.log('meteor beam', meteor_beam)
            self.log('bleu 1 beam', bleu1_beam)
            self.log('bleu 2 beam', bleu2_beam)
            self.log('bleu 3 beam', bleu3_beam)
            self.log('bleu 4 beam', bleu4_beam)
            self.log('rouge beam', rouge_beam)

        
        img_feats = self.hypernet.image_encoder(image.float())
        caps_pred, _ = self.hypernet.captioner(img_feats, caps.long(), 1.0)
        caps_pred_list.append(caps_pred)
        bleu1, bleu2, bleu3, bleu4,  meteor, rouge = metric_score(caps, caps_pred_list, self.vocab, self.metrics)
        self.log('meteor', meteor)
        self.log('bleu 1', bleu1)
        self.log('bleu 2', bleu2)
        self.log('bleu 3', bleu3)
        self.log('bleu 4', bleu4)
        self.log('rouge', rouge)


if __name__ == "__main__":
    glove_path = "/cortex/users/cohenza4/glove.6B.200d.txt"
    data_path = 'data/conceptual.tsv'
    save_path = "/cortex/users/cohenza4/checkpoint/HN/debug/"
    # data
    with open("data/vocab.pkl", 'rb') as f:
        vocab = pickle.load(f)
    dataset = ConceptualCaptions(data_path, vocab, 4)
    lengths = [int(len(dataset)*0.85), int(len(dataset)*0.05),
               len(dataset) - (int(len(dataset)*0.85) + int(len(dataset)*0.05))]
    print("lengths = ", lengths)
    train_data, val_data, test_data = torch.utils.data.random_split(dataset, lengths)
    train_loader = DataLoader(train_data, batch_size=4, num_workers=2,
                            shuffle=False, collate_fn= collate_fn_cc)
    val_loader = DataLoader(val_data, batch_size=4, num_workers=2,
                            shuffle=False, collate_fn= collate_fn_cc)
    test_loader = DataLoader(test_data, batch_size=4, num_workers=2,
                            shuffle=False, collate_fn= collate_fn_cc)
    model = HyperNetCC(200, 200, 200, len(vocab), vocab, mixup=False)
    print('Loading GloVe Embedding')
    model.load_glove_emb(glove_path)
    print(model)
    wandb_logger = WandbLogger(save_dir='/cortex/users/cohenza4')
    lr_monitor_callback = pl.callbacks.LearningRateMonitor()
    checkpoint_callback = ModelCheckpoint(dirpath=save_path, monitor="val_loss with TF", save_top_k=1)
    print('Starting Training')
    trainer = pl.Trainer(gpus=[6], num_nodes=1, precision=32,
                         logger=wandb_logger,
                         #overfit_batches = 1,
                         check_val_every_n_epoch=1,
                         default_root_dir=save_path,
                         log_every_n_steps=20,
                         auto_lr_find=True,
                         callbacks=[lr_monitor_callback, checkpoint_callback],
                         #accelerator='ddp',
                         max_epochs=55,
                         gradient_clip_val=5.,
                         )
    trainer.tune(model, train_loader, val_loader)
    trainer.fit(model, train_loader, val_loader)

    

    