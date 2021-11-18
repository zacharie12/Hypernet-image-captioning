import torch
from torch import nn
import numpy as np
import pickle
import pandas as pd
import csv
import nltk
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from nltk.translate.meteor_score import single_meteor_score, meteor_score
from nltk.translate.bleu_score import SmoothingFunction
from rouge_metric import PyRouge

def flip_parameters_to_tensors(module):
    attr = []
    while bool(module._parameters):
        attr.append(module._parameters.popitem() )
    setattr(module, 'registered_parameters_name', [])

    for i in reversed(attr):
        setattr(module, i[0], torch.zeros(i[1].shape,requires_grad=True))
        module.registered_parameters_name.append(i[0])

    module_name = [k for k,v in module._modules.items()]

    for name in module_name:
        if name == 'embed':
            continue
        if name == 'fc_out':
            continue

        flip_parameters_to_tensors(module._modules[name])

def set_all_parameters(module, theta):
    count = 0  

    for name in module.registered_parameters_name:
        # filter our FC and Embedding Weights 
        # only LSTMCell weights remain
        if name == 'weight':
            continue
        if name == 'bias':
            continue

        a = count
        b = a + getattr(module, name).numel()
        t = nn.Parameter(torch.reshape(theta[0,a:b], getattr(module, name).shape))
        setattr(module, name, t)

        count += getattr(module, name).numel()

    module_name = [k for k,v in module._modules.items()]
    for name in module_name:
        if name == 'embed':
            continue
        if name == 'fc_out':
            continue
        count += set_all_parameters(module._modules[name], theta)
    return count


class WordVectorLoader:

    def __init__(self, embed_dim):
        self.embed_index = {}
        self.embed_dim = embed_dim


    def load_glove(self, file_name):
        df = pd.read_csv(file_name, header=None, sep=' ', encoding='utf-8', quoting=csv.QUOTE_NONE)
        for index, row in df.iterrows():
            word = row[0]
            coefs = np.asarray(row[1:], dtype='float32')
            self.embed_index[word] = coefs
        try:
            self.embed_dim = len(coefs)
        except:
            pass



    def create_embedding_matrix(self, embeddings_file_name, word_to_index, max_idx, sep=' ', init='zeros', print_each=10000, verbatim=False):
        # Initialize embeddings matrix to handle unknown words
        if init == 'zeros':
            embed_mat = np.zeros((max_idx + 1, self.embed_dim))
        elif init == 'random':
            embed_mat = np.random.rand(max_idx + 1, self.embed_dim)
        else:
            raise Exception('Unknown method to initialize embeddings matrix')

        start = timeit.default_timer()
        with open(embeddings_file_name) as infile:
            for idx, line in enumerate(infile):
                elem = line.split(sep)
                word = elem[0]

                if verbatim is True:
                    if idx % print_each == 0:
                        print('[{}] {} lines processed'.format(datetime.timedelta(seconds=int(timeit.default_timer() - start)), idx), end='\r')

                if word not in word_to_index:
                    continue

                word_idx = word_to_index[word]

                if word_idx <= max_idx:
                    embed_mat[word_idx] = np.asarray(elem[1:], dtype='float32')


        if verbatim == True:
            print()

        return embed_mat


    def generate_embedding_matrix(self, word_to_index, max_idx, init='zeros'):
        # Initialize embeddings matrix to handle unknown words
        if init == 'zeros':
            embed_mat = np.zeros((max_idx + 1, self.embed_dim))
        elif init == 'random':
            embed_mat = np.random.rand(max_idx + 1, self.embed_dim)
        elif init == 'norm':
            embed_mat = np.random.normal(size=(max_idx+1, self.embed_dim))
        else:
            raise Exception('Unknown method to initialize embeddings matrix')

        for word, i in word_to_index.items():
            if i > max_idx:
                continue
            embed_vec = self.embed_index.get(word)
            if embed_vec is not None:
                embed_mat[i] = embed_vec

        return embed_mat


    def generate_centroid_embedding(self, word_list, avg=False):
        centroid_embedding = np.zeros((self.embed_dim, ))
        num_words = 0
        for word in word_list:
            if word in self.embed_index:
                num_words += 1
                centroid_embedding += self.embed_index.get(word)
        # Average embedding if needed
        if avg is True:
            if num_words > 0:
                centroid_embedding /= num_words
        return centroid_embedding


def cap_to_text(cap, voc, tokenized=False):
    sent = []
    argmaxed_cap = torch.argmax(cap, dim=1)
    for i in range(len(argmaxed_cap)):
        word = voc.i2w[argmaxed_cap[i].item()]
        if word == '<pad>' or word == '<s>' :
            continue
        if word == '</s>':
            break
        sent.append(word)
    if tokenized:
        return sent
    else:
        return " ".join(sent)


def cap_to_text_gt(cap, voc, tokenized=False):
    sent = []
    for i in range(len(cap)):
        word = voc.i2w[cap[i].item()]
        if word == '<pad>' or word == '<s>' :
            continue
        if word == '</s>':
            break
        sent.append(word)

    if tokenized:
        return sent
    else:
        return " ".join(sent)

def cap_to_text_gt_viz(cap, voc, tokenized=False):
    cap = cap[0]
    sent = []
    for i in range(len(cap)):
        word = voc.i2w[cap[i].item()]
        if word == '<pad>' or word == '<s>' :
            continue
        if word == '</s>':
            break
        sent.append(word)

    if tokenized:
        return sent
    else:
        return " ".join(sent)


def metric_score(gt_caps, pred_caps, vocab, metrics):

    tokenized_hyp_batch = []
    tokenized_ref_batch = []
    hyp_batch = []
    ref_batch = []
    #max_len  = max(len(pred_caps), len(gt_caps))
    for i in range(len(pred_caps)): 
        caps_pred_idx = torch.squeeze(pred_caps[i])
        gt_idx = torch.squeeze(gt_caps[i])

        tokenized_ref_text = cap_to_text_gt(gt_idx, vocab, tokenized=True)
        ref_text = cap_to_text_gt(gt_idx, vocab, tokenized=False)
        tokenized_hyp_text = cap_to_text(caps_pred_idx, vocab, tokenized=True)
        hyp_text = cap_to_text(caps_pred_idx, vocab, tokenized=False)

        hyp_batch.append(hyp_text)
        ref_batch.append(ref_text)
        tokenized_hyp_batch.append(tokenized_hyp_text)
        tokenized_ref_batch.append([tokenized_ref_text])
    output = []
    for metric in metrics:
        if metric.name == 'bleu':
            metric.add_batch(predictions=tokenized_hyp_batch, references=tokenized_ref_batch)
            output.append(metric.compute()['bleu'])
        elif metric.name == 'meteor':
            metric.add_batch(predictions=hyp_batch, references=ref_batch)
            output.append(metric.compute()['meteor'])
        elif metric.name == 'rouge':
            #metric.add_batch(predictions=hyp_batch, references=ref_batch)
            #output.append(metric.compute()['rougeL'][1][2])
            output.append(0.0)
    
    return output


def sample_multinomial_topk(distribution, k=10):
    distribution_k, idx = torch.topk(distribution, k)
    chosen = torch.multinomial(distribution_k, 1).t()[0]
    chosen_idx = torch.zeros([len(chosen), 1])
    for i in range(len(chosen)):
        chosen_idx[i] = idx[i][chosen[i]] 

        chosen_idx = chosen_idx.t()[0].int()
        return chosen_idx


def clean_sentence(output, voc):
    words = [voc.i2w.get(idx) for idx in output]
    words = [word for word in words if word not in ('<s>', ',', '<pad>', '</s>')]
    sentence = " ".join(words)
    return sentence