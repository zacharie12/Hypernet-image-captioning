import re
import pickle
from collections import Counter
import nltk


class Vocab:
    '''vocabulary'''
    def __init__(self):
        self.w2i = {}
        self.i2w = {}
        self.ix = 0

    def add_word(self, word):
        if word not in self.w2i:
            self.w2i[word] = self.ix
            self.i2w[self.ix] = word
            self.ix += 1

    def __call__(self, word):
        if word not in self.w2i:
            return self.w2i['<unk>']
        return self.w2i[word]

    def __len__(self):
        return len(self.w2i)


def build_vocab():
    '''build vocabulary'''
    # define vocabulary
    vocab = Vocab()
    # add special tokens
    vocab.add_word('<pad>')
    vocab.add_word('<s>')
    vocab.add_word('</s>')
    vocab.add_word('<unk>')
    vocab.add_word('factual')
    vocab.add_word('humorous')
    vocab.add_word('romantic')
    
    # add words
    path_train = 'data/train_cap_100.txt'
    path_val = 'data/val_cap_100.txt'
    path_test = 'data/test_cap_100.txt'
    captions_train = extract_captions(path_train)
    captions_val = extract_captions(path_val)
    captions_test = extract_captions(path_test)
    words_train = nltk.tokenize.word_tokenize(captions_train)
    words_val = nltk.tokenize.word_tokenize(captions_val)
    words_test = nltk.tokenize.word_tokenize(captions_test)
    words = words_train + words_val + words_test
    counter = Counter(words)
    words = [word for word, cnt in counter.items() if cnt >= 2]
    for word in words:
        vocab.add_word(word)

    return vocab


def extract_captions(path):
    '''extract captions from data files for building vocabulary'''
    text = ''
    with open(path, 'r') as f:
        res = f.readlines()

    r = re.compile(r'\d*.jpg#\d*')
    for line in res:
        line = r.sub('', line)
        line = line.replace('.', '')
        line = line.strip()
        sentence = line.split()
        sentence = sentence[1: len(sentence)-1]
        line = ' '.join(sentence)
        text += line + ' '


    return text.strip().lower()


if __name__ == '__main__':
    vocab = build_vocab()
    print(vocab.__len__())
    with open('data/vocab_CC.pkl', 'wb') as f:
        pickle.dump(vocab, f)