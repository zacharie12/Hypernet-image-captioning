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



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)

class Flickr7kDataset(Dataset):
    '''Flickr7k dataset'''
    def __init__(self, img_dir, caption_file, vocab, transform=None, get_path=False, multiple_image=False, batch_size=8):
        '''
        Args:
            img_dir: Direcutory with all the images
            caption_file: Path to the factual caption file
            vocab: Vocab instance
            transform: Optional transform to be applied
        '''
        self.img_dir = img_dir
        self.imgname_caption_list = self._get_imgname_and_caption(caption_file)
        self.vocab = vocab
        self.transform = transform
        self.get_path = get_path
        self.multiple_image = multiple_image
        self.batch_size = batch_size

    def _get_imgname_and_caption(self, caption_file):
        '''extract image name and caption from factual caption file'''
        with open(caption_file, 'r') as f:
            res = f.readlines()

        imgname_caption_list = []
        r = re.compile(r'#\d*')
        for line in res:
            img_and_cap = r.split(line)
            img_and_cap = [x.strip() for x in img_and_cap]
            imgname_caption_list.append(img_and_cap)

        return imgname_caption_list

    def __len__(self):
        return len(self.imgname_caption_list) // 5

    def __getitem__(self, ix):
        '''return one data pair (image and captioin)'''
        idx = ix*5 + random.randint(0, 4)
        img_name = self.imgname_caption_list[idx][0]
        img_name = os.path.join(self.img_dir, img_name)
        caption = self.imgname_caption_list[idx][1]

        image = skimage.io.imread(img_name)
        if self.transform is not None:
            image = self.transform(image)

        # convert caption to word ids
        r = re.compile("\.")
        tokens = nltk.tokenize.word_tokenize(r.sub("", caption).lower())
        caption = []
        caption.append(self.vocab('<s>'))
        caption.extend([self.vocab(token) for token in tokens])
        caption.append(self.vocab('</s>'))
        caption = torch.Tensor(caption)
        if self.multiple_image:
            caption_list = []
            for i in range(self.batch_size):
                idximage = random.choice(range(1, idximage*5) + range(idximage*5+1, 7000))
                idxcaption = idximage*5 + random.randint(0, 4)
                caption_helper = self.imgname_caption_list[idxcaption][1]

                # convert caption to word ids
                r = re.compile("\.")
                tokens = nltk.tokenize.word_tokenize(r.sub("", caption_helper).lower())
                caption_list[i].append(self.vocab('<s>'))
                caption_list[i].extend([self.vocab(token) for token in tokens])
                caption_list[i].append(self.vocab('</s>'))
                caption_list[i] = torch.Tensor(caption_list[i])

        
        if self.get_path:
            return image, caption, img_name
        elif self.multiple_image:
            return image, caption, caption_list
        else:
            return image, caption


class FlickrStyle7kDataset(Dataset):
    '''Styled caption dataset'''
    def __init__(self, caption_file, vocab, multiple_image=False, batch_size=8):
        '''
        Args:
            caption_file: Path to styled caption file
            vocab: Vocab instance
        '''
        self.caption_list = self._get_caption(caption_file)
        self.vocab = vocab
        self.multiple_image = multiple_image
        self.batch_size = batch_size

    def _get_caption(self, caption_file):
        '''extract caption list from styled caption file'''
        with open(caption_file, 'r') as f:
            caption_list = f.readlines()

        caption_list = [x.strip() for x in caption_list]
        return caption_list

    def __len__(self):
        return len(self.caption_list)

    def __getitem__(self, ix):
        caption = self.caption_list[ix]
        # convert caption to word ids
        r = re.compile("\.")
        tokens = nltk.tokenize.word_tokenize(r.sub("", caption).lower())
        caption = []
        caption.append(self.vocab('<s>'))
        caption.extend([self.vocab(token) for token in tokens])
        caption.append(self.vocab('</s>'))
        caption = torch.Tensor(caption)
        if self.multiple_image:
            caption_list = []
            for i in range(self.batch_size):
                idxcaption = random.choice(range(1, ix) + range(ix+1, 6))
                caption_helper = self.caption_list[idxcaption]
                # convert caption to word ids
                r = re.compile("\.")
                tokens = nltk.tokenize.word_tokenize(r.sub("", caption_helper).lower())
                caption_list[i].append(self.vocab('<s>'))
                caption_list[i].extend([self.vocab(token) for token in tokens])
                caption_list[i].append(self.vocab('</s>'))
                caption_list[i] = torch.Tensor(caption_list[i])

        if self.multiple_image:
            return caption, caption_list
        else:    
            return caption



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


def get_dataset(img_dir, caption_file, vocab, dataset="CC", transform=None, get_path=False, multiple_image=False, batch_size=8):
    '''Return data_loader'''
    if transform is None:
        transform = transforms.Compose([
            Rescale((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
    if dataset == "Fliker factual":
        data = Flickr7kDataset(img_dir, caption_file, vocab, transform, get_path, multiple_image=False, batch_size=8)
    elif dataset == "Fliker style":
        data = FlickrStyle7kDataset(caption_file, vocab, multiple_image=False, batch_size=8)
    else:
        data = ConceptualCaptions(img_dir, caption_file, vocab, transform, get_path, batch_size=8)
    return data


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


def collate_fn_cc(data):
    '''create minibatch tensors from data(list of tuple(image, caption, domain))'''
    images, captions, domain = zip(*data)

    # images : tuple of 3D tensor -> 4D tensor
    images = torch.stack(images, 0)

    # captions : tuple of 1D Tensor -> 2D tensor
    lengths = torch.LongTensor([len(cap) for cap in captions])
    captions = [pad_sequence(cap, max(lengths)) for cap in captions]
    captions = torch.stack(captions, 0)

    return images, captions, lengths, domain 

def collate_fn(data):
    '''create minibatch tensors from data(list of tuple(image, caption))'''
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)


    # images : tuple of 3D tensor -> 4D tensor
    images = torch.stack(images, 0)

    # captions : tuple of 1D Tensor -> 2D tensor
    lengths = torch.LongTensor([len(cap) for cap in captions])
    captions = [pad_sequence(cap, max(lengths)) for cap in captions]
    captions = torch.stack(captions, 0)

    return images, captions, lengths

def collate_fn_styled(captions):
    captions.sort(key=lambda x: len(x), reverse=True)

    # tuple of 1D Tensor -> 2D Tensor
    lengths = torch.LongTensor([len(cap) for cap in captions])
    captions = [pad_sequence(cap, max(lengths)) for cap in captions]
    captions = torch.stack(captions, 0)

    return captions, lengths

def combine_collate_fn(data):
    # create minibatch from list of [tuple(img, cap), cap, cap]
    factual_data = [d[0] for d in data]
    humour_data = [d[1] for d in data]
    romantic_data = [d[2] for d in data]
    cc_data = [d[3] for d in data]
    imgs, *collated_factual = collate_fn(factual_data)
    collated_humour = collate_fn_styled(humour_data)
    collated_romantic = collate_fn_styled(romantic_data)
    cc_imgs, captions, lengths, domain  = collate_fn_cc(cc_data)
    dataset = random.choice(["CC","Fliker"])
    if dataset == "Fliker":
        domain, caps = random.choice([('factual', tuple(collated_factual)), ('humour', collated_humour), ('romantic', collated_romantic)])
        captions, lengths = caps[0], caps[1]
    else:
        imgs = cc_imgs
    return imgs, captions, lengths, domain

def collate_fn_test(data, style='CC'):
    # create minibatch from list of [tuple(img, cap), cap, cap]
    factual_data = [d[0] for d in data]
    humour_data = [d[1] for d in data]
    romantic_data = [d[2] for d in data]
    cc_data = [d[3] for d in data]
    zero_shot_data = [d[4] for d in data]

    imgs, *collated_factual = collate_fn(factual_data)
    collated_humour = collate_fn_styled(humour_data)
    collated_romantic = collate_fn_styled(romantic_data)
    cc_imgs, captions, lengths, domain = collate_fn_cc(cc_data)
    zero_shot_imgs, zero_shot_captions, zero_shot_lengths, zero_shot_domain = collate_fn_cc(zero_shot_data)

    if style == 'factual':
        domain, caps = random.choice([('factual', tuple(collated_factual))])
        captions, lengths = caps[0], caps[1]
    elif style == 'humour':
        domain,caps = random.choice([('humour', collated_humour)])
        captions, lengths = caps[0], caps[1]
    elif style == 'romantic':
        domain,caps = random.choice([('romantic', collated_romantic)])  
        captions, lengths = caps[0], caps[1]
    elif style == 'CC':
        imgs = cc_imgs
    else:
        imgs, captions, lengths, domain = zero_shot_imgs, zero_shot_captions, zero_shot_lengths, zero_shot_domain

    return imgs, captions, lengths, domain

def main():
    img_dir_fliker = 'data/flickr7k_images'
    img_dir_CC = 'data/200_conceptual_images_val'
    caption_CC= 'data/test_cap_100.txt'
    caption_fac= 'data/fac_cap_test.txt'
    caption_hum= 'data/humor_cap_test.txt'
    caption_rom= 'data/rom_cap_test.txt'
    with open("data/vocab.pkl", 'rb') as f:
        vocab = pickle.load(f)

    data_fac = get_dataset(img_dir_fliker, caption_fac, vocab, "Fliker factual")
    data_hum = get_dataset(img_dir_fliker, caption_hum, vocab, "Fliker style")
    data_rom= get_dataset(img_dir_fliker, caption_rom, vocab, "Fliker style")
    data_CC = get_dataset(img_dir_CC, caption_CC, vocab)
    data_concat = ConcatDataset(data_fac, data_hum, data_rom, data_CC)
    train_loader = DataLoader(data_concat, batch_size=8, num_workers=2,
                            shuffle=False, collate_fn=lambda x: collate_fn_test(x, 'romantic'))
    for images, captions, lengths, domain in train_loader:                     
        print("done")



if __name__ == '__main__':
    main()

