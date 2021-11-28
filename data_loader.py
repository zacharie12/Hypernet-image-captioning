import os
import re
import pickle
import nltk
import skimage.io
import skimage.transform
import torch
import random
from torch.nn.functional import pad
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from build_vocab import Vocab

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)
    

class FlickrCombiner(torch.utils.data.Dataset):
    def __init__(self, orig_data, styled_data):
        self.orig_data = orig_data
        self.styled_data = styled_data
    
    def __getitem__(self, i):
        img, fac_cap = self.orig_data[i]
        styled_caps = [('factual', fac_cap)]
        for (style_name, styled_dataset) in self.styled_data:
            styled_caps.append((style_name, styled_dataset[i]))

        return img, random.choice(styled_caps)
        
    def __len__(self):
        return min(min(len(d[1]) for d in self.styled_data), len(self.orig_data))


class Flickr7kDataset(Dataset):
    '''Flickr7k dataset'''
    def __init__(self, img_dir, caption_file, vocab, transform=None, get_path=False, all_caption=False):
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
        self.all_caption = all_caption

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

        all_caption = []
        if self.all_caption:
            for i in range(5):
                idx = ix*5 + i
            caption = self.imgname_caption_list[idx][1]

            image = skimage.io.imread(img_name)
            if self.transform is not None:
                image = self.transform(image)

            # convert caption to word ids
            r = re.compile("\.")
            tokens = nltk.tokenize.word_tokenize(r.sub("", caption).lower())
            caption_i = []
            caption_i.append(self.vocab('<s>'))
            caption_i.extend([self.vocab(token) for token in tokens])
            caption_i.append(self.vocab('</s>'))
            caption_i = torch.Tensor(caption_i)
            all_caption.append(caption_i)
        if self.get_path:
            return image, caption, img_name
        elif self.all_caption:
            image, caption, all_caption
        else:
            return image, caption


class FlickrStyle7kDataset(Dataset):
    '''Styled caption dataset'''
    def __init__(self, caption_file, vocab):
        '''
        Args:
            caption_file: Path to styled caption file
            vocab: Vocab instance
        '''
        self.caption_list = self._get_caption(caption_file)
        self.vocab = vocab

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
        return caption


def get_dataset(img_dir, caption_file, vocab,
                    transform=None, get_path=False):
    '''Return data_loader'''
    if transform is None:
        transform = transforms.Compose([
            Rescale((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    flickr7k = Flickr7kDataset(img_dir, caption_file, vocab, transform, get_path)

    return flickr7k

def get_styled_dataset(caption_file, vocab):
    '''Return data_loader for styled caption'''
    flickr_styled_7k = FlickrStyle7kDataset(caption_file, vocab)
    
    return flickr_styled_7k


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


def collate_fn_visualize(data):
    '''create minibatch tensors from data(list of tuple(image, caption))'''
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, img_path = zip(*data)

    # images : tuple of 3D tensor -> 4D tensor
    images = torch.stack(images, 0)

    # captions : tuple of 1D Tensor -> 2D tensor
    lengths = torch.LongTensor([len(cap) for cap in captions])
    captions = [pad_sequence(cap, max(lengths)) for cap in captions]
    captions = torch.stack(captions, 0)

    return images, captions, lengths, img_path


def collate_fn_styled(captions):
    captions.sort(key=lambda x: len(x), reverse=True)

    # tuple of 1D Tensor -> 2D Tensor
    lengths = torch.LongTensor([len(cap) for cap in captions])
    captions = [pad_sequence(cap, max(lengths)) for cap in captions]
    captions = torch.stack(captions, 0)

    return captions, lengths


def pad_sequence(seq, max_len):
    seq = torch.cat((seq, torch.zeros(max_len - len(seq))))
    return seq

def flickr_collate_fn(data):
    # create minibatch from list of [tuple(img, cap), cap, cap]
    factual_data = [d[0] for d in data]
    humour_data = [d[1] for d in data]
    romantic_data = [d[2] for d in data]

    imgs, *collated_factual = collate_fn(factual_data)
    collated_humour = collate_fn_styled(humour_data)
    collated_romantic = collate_fn_styled(romantic_data)

    caps = random.choice([('factual', tuple(collated_factual)), ('humour', collated_humour), ('romantic', collated_romantic)])
    return imgs, caps

def collate_fn_classifier(data):

    factual_data = [d[0] for d in data]
    humour_data = [d[1] for d in data]
    romantic_data = [d[2] for d in data]
    
    imgs, *collated_factual = collate_fn(factual_data)
    collated_humour = collate_fn_styled(humour_data)
    collated_romantic = collate_fn_styled(romantic_data)
    caps = random.choice([('factual', tuple(collated_factual)), ('humour', collated_humour), ('romantic', collated_romantic)])
    style = caps[0]
    if style == "factual":
        _label = torch.tensor([[1.0, 0.0,0.0]])
    elif style == "humour":
        _label = torch.tensor([[0.0, 1.0,0.0]])
    else:
        _label = torch.tensor([[0.0, 0.0,1.0]])        

    label_list, text_list, offsets = [], [], [0]
    for (style, (processed_text, lengths)) in caps:
        label_list.append(_label)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list.to(device), text_list.to(device), offsets.to(device)




def flickr_collate_fn_visualize(data):
    # create minibatch from list of [tuple(img, cap), cap, cap]
    factual_data = [d[0] for d in data]
    humour_data = [d[1] for d in data]
    romantic_data = [d[2] for d in data]

    imgs, *collated_factual, img_path = collate_fn_visualize(factual_data)
    collated_humour = collate_fn_styled(humour_data)
    collated_romantic = collate_fn_styled(romantic_data)
    return ((imgs, collated_factual, img_path), collated_humour, collated_romantic) 

def flickr_collate_style(data, style='romantic'):
    factual_data = [d[0] for d in data]
    humour_data = [d[1] for d in data]
    romantic_data = [d[2] for d in data]

    imgs, *collated_factual = collate_fn(factual_data)
    collated_humour = collate_fn_styled(humour_data)
    collated_romantic = collate_fn_styled(romantic_data)

    if style == 'factual':
        caps = random.choice([('factual', tuple(collated_factual))])
    elif style == 'humour':
        caps = random.choice([('humour', collated_humour)])
    else:
        caps = random.choice([('romantic', collated_romantic)])        
    return imgs, caps    

if __name__ == "__main__":
    with open("data/vocab.pkl", 'rb') as f:
        vocab = pickle.load(f)

    img_path = "data/flickr7k_images"
    cap_path = "data/factual_train.txt"
    cap_path_humor = "data/humor/funny_train.txt"
    cap_path_romantic = "data/romantic/romantic_train.txt"
    orig_dataset = get_dataset(img_path, cap_path, vocab, get_path=False)
    humor_dataset = get_styled_dataset(cap_path_humor, vocab)
    romantic_dataset = get_styled_dataset(cap_path_romantic, vocab)
    data_concat = ConcatDataset(orig_dataset, humor_dataset, romantic_dataset)
    lengths = [int(len(data_concat)*0.8),
               len(data_concat) - int(len(data_concat)*0.8)]
    train_data, val_data = torch.utils.data.random_split(data_concat, lengths)
    train_loader = DataLoader(train_data, batch_size=1, num_workers=1,
                              shuffle=False, collate_fn=flickr_collate_fn_visualize)
    val_loader = DataLoader(val_data, batch_size=1, num_workers=1,
                            shuffle=False, collate_fn=flickr_collate_fn_visualize)

    #for ((img_tensor, fac_cap, img_name), hum_cap, rom_cap), (img_tensor, fac_cap) in [(val_data), (val_loader)]:
    for ((img_tensor, fac_cap, img_name), (hum_cap, lengths), (rom_cap, lengths)) in val_loader:
        print("fac_cap ")
        
        
