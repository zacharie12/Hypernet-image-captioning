from collections import Counter
import tldextract
from PIL import Image
import requests
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import skimage.transform
from torchvision import transforms
import torch
import re
import os
import pickle
import nltk
import skimage.io
import random
from torch.nn.functional import pad
from torch.utils.data import Dataset, DataLoader
from build_vocab import Vocab
from matplotlib import cm



class ConceptualCaptions(torch.utils.data.Dataset):
    def __init__(self, img_dir, caption_file, vocab, transform=None, get_path=False, batch_size=8):
        self.img_dir = img_dir
        self.imgname_caption_list_domain = self._get_imgname_and_caption_and_domain(caption_file)
        self.vocab = vocab
        self.transform = transform
        self.get_path = get_path
        self.batch_size = batch_size
        self.current_domain = None
        self.counter = batch_size
        self.curr_list = []
        self.range_domain = self._get_range_domain(caption_file)



    def _get_range_domain(self, caption_file):
        start, curr = 0, 0 
        dict_range_domain = {}
        prev_domain = None
        with open(caption_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                curr += 1
                x = line.split("     ")
                domain = x[2].replace("\n", '')
                if prev_domain == None:
                    prev_domain =domain
                elif prev_domain != domain:
                    dict_range_domain[prev_domain] = [start, curr]
                    start = curr
                    prev_domain =domain
        dict_range_domain[domain] = [start, curr-1]
        return dict_range_domain

    def _get_imgname_and_caption_and_domain(self, caption_file):
        '''extract image name and caption from factual caption file'''
        ids, captions, domains = [], [], []
        with open(caption_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                x = line.split("     ")
                ids.append(x[0])
                captions.append(x[1])
                domain = x[2].replace("\n", '')
                domains.append(domain)
        imgname_caption_list_domain = [ids, captions, domains]
        return imgname_caption_list_domain

    def __len__(self):
        return len(self.imgname_caption_list_domain[0])

    def __getitem__(self, idx):
        domain = self.imgname_caption_list_domain[2][idx]
        if  self.current_domain != None:
            if len(self.curr_list) == 0:
                self.curr_list = list(range(self.range_domain[self.current_domain][0], self.range_domain[self.current_domain][1]))
            idx = random.sample(self.curr_list, 1)[0]
            self.curr_list.remove(idx)
            domain = self.imgname_caption_list_domain[2][idx]
        else:
            self.current_domain = domain
            self.curr_list = list(range(self.range_domain[self.current_domain][0], self.range_domain[self.current_domain][1]))
        self.counter -= 1
        img_name = self.imgname_caption_list_domain[0][idx]
        img_name = os.path.join(self.img_dir, img_name)
        caption = self.imgname_caption_list_domain[1][idx]    
        image = skimage.io.imread(img_name)
        
        if len(image.shape) == 2:
            try:
                colmap = cm.get_cmap('viridis', 256)
                np.savetxt('cmap.csv', (colmap.colors[...,0:3]*255).astype(np.uint8), fmt='%d', delimiter=',')
                lut = np.genfromtxt('cmap.csv', dtype=np.uint8, delimiter=',')
                result = np.zeros((*image.shape,3), dtype=np.uint8)
                np.take(lut, image, axis=0, out=result)
                image = Image.fromarray(result)
                image = np.array(image)
            except ValueError:                
                if  self.current_domain != None:
                    if len(self.curr_list) == 0:
                        self.curr_list = list(range(self.range_domain[self.current_domain][0], self.range_domain[self.current_domain][1]))
                    #self.curr_list.remove(idx)
                    idx = random.sample(self.curr_list, 1)[0]
                    self.curr_list.remove(idx)
                    domain = self.imgname_caption_list_domain[2][idx]
                else:
                    self.current_domain = domain
                    self.curr_list = list(range(self.range_domain[self.current_domain][0], self.range_domain[self.current_domain][1]))
                img_name = self.imgname_caption_list_domain[0][idx]
                img_name = os.path.join(self.img_dir, img_name)
                caption = self.imgname_caption_list_domain[1][idx]    
                image = skimage.io.imread(img_name)
        try:
            if self.transform is not None:
                image = self.transform(image)
        except RuntimeError:
            if self.current_domain != None:
                if len(self.curr_list) == 0:
                    self.curr_list = list(range(self.range_domain[self.current_domain][0], self.range_domain[self.current_domain][1]))
                #self.curr_list.remove(idx)
                idx = random.sample(self.curr_list, 1)[0]
                self.curr_list.remove(idx)
                domain = self.imgname_caption_list_domain[2][idx]
            else:
                self.current_domain = domain
                self.curr_list = list(range(self.range_domain[self.current_domain][0], self.range_domain[self.current_domain][1]))
            img_name = self.imgname_caption_list_domain[0][idx]
            img_name = os.path.join(self.img_dir, img_name)
            caption = self.imgname_caption_list_domain[1][idx]    
            image = skimage.io.imread(img_name)
            if self.transform is not None:
                try:
                    image = self.transform(image)
                except RuntimeError:
                    if self.current_domain != None:
                        if len(self.curr_list) == 0:
                            self.curr_list = list(range(self.range_domain[self.current_domain][0], self.range_domain[self.current_domain][1]))
                        #self.curr_list.remove(idx)
                        idx = random.sample(self.curr_list, 1)[0]
                        self.curr_list.remove(idx)
                        domain = self.imgname_caption_list_domain[2][idx]
                    else:
                        self.current_domain = domain
                        self.curr_list = list(range(self.range_domain[self.current_domain][0], self.range_domain[self.current_domain][1]))
                    img_name = self.imgname_caption_list_domain[0][idx]
                    img_name = os.path.join(self.img_dir, img_name)
                    caption = self.imgname_caption_list_domain[1][idx]    
                    image = skimage.io.imread(img_name)
                    if self.transform is not None:
                        image = self.transform(image)
    
        if self.counter == 0:
            self.current_domain = None
            self.counter = self.batch_size
            self.curr_list = []

        r = re.compile("\.")
        tokens = nltk.tokenize.word_tokenize(r.sub("", caption).lower())
        caption = []
        caption.append(self.vocab('<s>'))
        caption.extend([self.vocab(token) for token in tokens])
        caption.append(self.vocab('</s>'))
        caption = torch.Tensor(caption)
        if self.get_path:
            return image, caption, domain, img_name
        else:
            return image, caption, domain
       

def get_dataset(img_dir, caption_file, vocab, transform=None, get_path=False, batch_size=8):
    '''Return data_loader'''
    if transform is None:
        transform = transforms.Compose([
            Rescale((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
    cc = ConceptualCaptions(img_dir, caption_file, vocab, transform, get_path, batch_size=8)
    return cc

class Rescale:
    '''Rescale the image to a given size
    Args:
        output_size(int or tuple)
    '''
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image):
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        image = skimage.transform.resize(image, (new_h, new_w))

        return image

def pad_sequence(seq, max_len):
    seq = torch.cat((seq, torch.zeros(max_len - len(seq))))
    return seq



def collate_fn(data):
    '''create minibatch tensors from data(list of tuple(image, caption, domain))'''
    images, captions, domain = zip(*data)

    # images : tuple of 3D tensor -> 4D tensor
    images = torch.stack(images, 0)

    # captions : tuple of 1D Tensor -> 2D tensor
    lengths = torch.LongTensor([len(cap) for cap in captions])
    captions = [pad_sequence(cap, max(lengths)) for cap in captions]
    captions = torch.stack(captions, 0)

    return images, captions, lengths, domain        
    


def main():
    img_dir = 'data/conceptual_images_train'
    caption_file = 'data/CC_train.txt'
    with open("data/vocab.pkl", 'rb') as f:
        vocab = pickle.load(f)
    train_data = get_dataset(img_dir, caption_file, vocab)
    train_loader = DataLoader(train_data, batch_size=8, num_workers=2,
                            shuffle=False, collate_fn= collate_fn)
    for images, captions, lengths, domain in train_loader:                     
        print("done")



if __name__ == '__main__':
    main()