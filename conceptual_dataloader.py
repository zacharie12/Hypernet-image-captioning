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



class ConceptualCaptions(torch.utils.data.Dataset):
    def __init__(self, path, vocab, batch_size=8):
        self.data_path = path
        self.urls = []
        self.captions = []
        self.domains = []
        self.vocab = vocab
        self.batch_size = batch_size
        with open(self.data_path , 'r') as f:
            lines = f.readlines()
            for line in lines:
                self.urls.append(line.split('\t')[1]) 
                self.captions.append(line.split('\t')[0])
                self.domains.append(tldextract.extract(line.split('\t')[1])[1])

    def __getitem__(self, i):
        url, caption, domain = self.urls[i], self.captions[i], self.domains[i]
        r = re.compile("\.")
        tokens = nltk.tokenize.word_tokenize(r.sub("", caption).lower())
        caption = []
        caption.append(self.vocab('<s>'))
        caption.extend([self.vocab(token) for token in tokens])
        caption.append(self.vocab('</s>'))
        caption = torch.Tensor(caption)
        return url, caption, domain 
        
    def __len__(self):
        return len(self.urls)


def pad_sequence(seq, max_len):
    seq = torch.cat((seq, torch.zeros(max_len - len(seq))))
    return seq

def collate_fn_cc(data):
    # create minibatch from list of [tuple(img, cap), cap, cap]
    url, captions, domain = [], [], []
    for i in range (len(data)):
        url.append(data[i][0])
        captions.append(data[i][1])
        domain.append(data[i][2])
    lengths = torch.LongTensor([len(cap) for cap in captions])
    captions = [pad_sequence(cap, max(lengths)) for cap in captions]
    captions = torch.stack(captions, 0)
    return url, captions, lengths, domain

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
def main():
    file_path = '/cortex/data/images/conceptual_captions/Train_GCC-training.tsv'
    out_path = 'data/conceptual_domains.tsv'
    with open("data/vocab.pkl", 'rb') as f:
        vocab = pickle.load(f)
    dataset = ConceptualCaptions(file_path, vocab, 4)
    lengths = [int(len(dataset)*0.8), int(len(dataset)*0.1),
               len(dataset) - (int(len(dataset)*0.8) + int(len(dataset)*0.1))]
    train_data, val_data, test_data = torch.utils.data.random_split(dataset, lengths)
    train_loader = DataLoader(train_data, batch_size=4, num_workers=2,
                            shuffle=False, collate_fn= collate_fn_cc)
    print("done")
















    '''
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            caption = line.split('\t')[0]
            url = line.split('\t')[1]
            domain = tldextract.extract(url)[1]  # get domain
            print("caption is ", caption)
            print("url is ", url)
            print("domain is ", domain)
            im = Image.open(requests.get(url, stream=True).raw)
            image = np.array(im)
            image = skimage.transform.resize(image, (new_h, new_w))
            


    '''

if __name__ == '__main__':
    main()