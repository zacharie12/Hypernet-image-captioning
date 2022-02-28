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
import math
import pandas as pd 
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer 
from scipy.spatial import distance
from sklearn.manifold import TSNE
from math import log
from transformers import BertTokenizer
from cider import Cider
from ptbtokenizer import PTBTokenizer
from collections import defaultdict

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


def cap_to_text_gt_batch(cap, voc, tokenized=False):
    out = []
    for j in range(len(cap)):
        sent = []
        for i in range(len(cap[j])):
            caption = cap[j]
            word = voc.i2w[caption[i].item()]
            if word == '<pad>' or word == '<s>' :
                continue
            if word == '</s>':
                break
            sent.append(word)
        if tokenized:
            out.append(sent)
        else:
            out.append(" ".join(sent))
    return out


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
    Cider_scoreur = Cider()

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
            output.append(metric.compute(max_order=1)['bleu'])
            metric.add_batch(predictions=tokenized_hyp_batch, references=tokenized_ref_batch)
            output.append(metric.compute(max_order=2)['bleu'])
            metric.add_batch(predictions=tokenized_hyp_batch, references=tokenized_ref_batch)
            output.append(metric.compute(max_order=3)['bleu'])
            metric.add_batch(predictions=tokenized_hyp_batch, references=tokenized_ref_batch)
            output.append(metric.compute(max_order=4)['bleu'])
        elif metric.name == 'meteor':
            metric.add_batch(predictions=hyp_batch, references=ref_batch)
            output.append(metric.compute()['meteor'])
        elif metric.name == 'rouge':
            metric.add_batch(predictions=hyp_batch, references=ref_batch)
            output.append(metric.compute()['rougeL'][1][2])
            #output.append(0.0)        
    gts = defaultdict(list)
    res = []
    for i in range(len(tokenized_hyp_batch)):
        #gts[i].append({"caption": tokenized_ref_text[i]})
        #res[i].append({"caption": tokenized_hyp_text[i]})
        double_list_gt = tokenized_ref_batch[i]
        list_gt = double_list_gt[0]
        str_gt = ' '.join(list_gt)
        str_hyp = ' '.join(tokenized_hyp_batch[i])
        gts[i].append(str_gt)
        res.append({'image_id': i, 'caption': [str_hyp]})
    score, scores = Cider_scoreur.compute_score(gts, res)
    output.append(score)
        
    return output

def metric_score_test(gt_caps, pred_caps, vocab, metrics):
    Cider_scoreur = Cider()
    tokenized_hyp_batch = []
    tokenized_ref_batch = []
    hyp_batch = []
    ref_batch = []
    
    caps_pred_idx = pred_caps
    tokenized_hyp_text = cap_to_text_gt(caps_pred_idx, vocab, tokenized=True)
    hyp_text = cap_to_text_gt(caps_pred_idx, vocab, tokenized=False)
    hyp_batch.append(hyp_text)
    tokenized_hyp_batch.append(tokenized_hyp_text)

    for i in range(len(gt_caps)): 
        gt_idx = torch.squeeze(gt_caps[i])
        tokenized_ref_text = cap_to_text_gt(gt_idx, vocab, tokenized=True)
        ref_text = cap_to_text_gt(gt_idx, vocab, tokenized=False)
        ref_batch.append(ref_text)
        tokenized_ref_batch.append([tokenized_ref_text])    
    output = []
    for metric in metrics:
        if metric.name == 'bleu':
            metric.add_batch(predictions=tokenized_hyp_batch, references=tokenized_ref_batch)
            output.append(metric.compute(max_order=1)['bleu'])
            metric.add_batch(predictions=tokenized_hyp_batch, references=tokenized_ref_batch)
            output.append(metric.compute(max_order=2)['bleu'])
            metric.add_batch(predictions=tokenized_hyp_batch, references=tokenized_ref_batch)
            output.append(metric.compute(max_order=3)['bleu'])
            metric.add_batch(predictions=tokenized_hyp_batch, references=tokenized_ref_batch)
            output.append(metric.compute(max_order=4)['bleu'])
        elif metric.name == 'meteor':
            metric.add_batch(predictions=hyp_batch, references=ref_batch)
            output.append(metric.compute()['meteor'])
        elif metric.name == 'rouge':
            metric.add_batch(predictions=hyp_batch, references=ref_batch)
            output.append(metric.compute()['rougeL'][1][2])
    gts = defaultdict(list)
    res = []
    for i in range(len(tokenized_hyp_batch)):
        #gts[i].append({"caption": tokenized_ref_text[i]})
        #res[i].append({"caption": tokenized_hyp_text[i]})
        double_list_gt = tokenized_ref_batch[i]
        list_gt = double_list_gt[0]
        str_gt = ' '.join(list_gt)
        str_hyp = ' '.join(tokenized_hyp_batch[i])
        gts[i].append(str_gt)
        res.append({'image_id': i, 'caption': [str_hyp]})
    score, scores = Cider_scoreur.compute_score(gts, res)
    output.append(score)
    
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

#create list of different domain in two files
def get_domain_list(cap_dir1, cap_dir2):
    domains = []
    with open(cap_dir1, 'r') as f:
        lines = f.readlines()
        for line in lines:
            x = line.split("     ")
            domains.append(x[2])
    if len(cap_dir2) !=0:
        with open(cap_dir2, 'r') as f:
            lines = f.readlines()
            for line in lines:
                x = line.split("     ")
                domains.append(x[2])
    domains = list(dict.fromkeys(domains))
    return domains

# create dict where the keys are the domains in file txt and the value are histograme of num of apparance of each word in each domain
def get_hist_embedding(cap_dir1, vocab, list_domain, do_log = True):
    counter_per_domain = {}
    eps = 0.0001
    for cur_domain in list_domain:
        counter_word = [0]*(len(vocab)+1)
        with open(cap_dir1, 'r') as f:
            lines = f.readlines()
            for line in lines:
                x = line.split("     ")
                domain = x[2]
                if cur_domain == domain:
                    cap = x[1].split(" ")
                    for word in cap:
                        try:
                            counter_word[vocab.w2i[word]] +=1
                        except KeyError:
                            counter_word[len(vocab)] +=1
        if do_log:
            for i in range(len(counter_word)):
                counter_word[i] = log(counter_word[i]+eps,10) 
        counter_per_domain[cur_domain.replace("\n", '')] = counter_word
    return counter_per_domain

def tfidf_hist(cap_dir1, vocab, list_domain):
    counter_word = [0]*(len(vocab)+1)
    tfidf_perdomain ={}
    str_domain = ''
    docs = []
    for cur_domain in list_domain:
        with open(cap_dir1, 'r') as f:
            lines = f.readlines()
            for line in lines:
                x = line.split("     ")
                domain = x[2]
                if cur_domain == domain:
                    str_domain += x[1]
        docs.append(str_domain)
    cv = CountVectorizer() 
    word_count_vector = cv.fit_transform(docs)
    tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True) 
    tfidf_transformer.fit(word_count_vector)
    # count matrix 
    count_vector = cv.transform(docs) 
    # tf-idf scores 
    tf_idf_vector = tfidf_transformer.transform(count_vector)
    for i in range(len(list_domain)):
        dom_i = tf_idf_vector[i].T.todense().tolist()
        flat_dom_i= [item for sublist in dom_i for item in sublist]
        tfidf_perdomain[list_domain[i].replace("\n", '')] = flat_dom_i
    return tfidf_perdomain


def get_jsd_tsne(cap_dir1, vocab, list_domain, num_domain, n_tsne, zero_shot=False, list_zeroshot=[]):
    tsne_domain = {}
    test_file = 'data/one_shot_captions.txt'
    counter_per_domain = get_hist_embedding(cap_dir1, vocab, list_domain, False)
    if zero_shot:
        dict_zero_shot = get_hist_embedding(test_file, vocab, list_zeroshot, False)
        counter_per_domain.update(dict_zero_shot)
    domain_list = list(counter_per_domain.keys())
    domains_hist = list(counter_per_domain.values())
    mat_dist = np.zeros((num_domain, num_domain))
    for row in range(num_domain):
        for col in range(num_domain):
            a, b = domains_hist[row], domains_hist[col]
            mat_dist[row][col] =  distance.jensenshannon(a, b)
            mat_dist = np.nan_to_num(mat_dist)
    x_tsne = TSNE(n_components=n_tsne, init='random').fit_transform(mat_dist)
    for i in range(len(domain_list)):
        tsne_domain[domain_list[i]] = [x_tsne[:, 0][i], x_tsne[:, 1][i]]
    return tsne_domain
    


class CustomBertTokenizer(BertTokenizer):
    def __init__(self, *args, **kwargs):
        super(CustomBertTokenizer, self).__init__(*args, **kwargs)

    def decode(self, token_ids, skip_special_tokens=True,
               clean_up_tokenization_spaces=True, end_flags=[]):
        filtered_tokens = self.convert_ids_to_tokens(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            end_flags=end_flags)

        # To avoid mixing byte-level and unicode for byte-level BPT
        # we need to build string separatly for added tokens and byte-level tokens
        # cf. https://github.com/huggingface/transformers/issues/1133
        sub_texts = []
        current_sub_text = []
        for token in filtered_tokens:
            if skip_special_tokens and token in self.all_special_ids:
                continue
            if token in self.added_tokens_encoder:
                if current_sub_text:
                    sub_texts.append(self.convert_tokens_to_string(current_sub_text))
                    current_sub_text = []
                sub_texts.append(" " + token)
            else:
                current_sub_text.append(token)
        if current_sub_text:
            sub_texts.append(self.convert_tokens_to_string(current_sub_text))
        text = ''.join(sub_texts)

        if clean_up_tokenization_spaces:
            clean_text = self.clean_up_tokenization(text)
            return clean_text
        else:
            return text

    def convert_ids_to_tokens(self, ids, skip_special_tokens=False, end_flags=[]):
        if isinstance(ids, int):
            if ids in self.added_tokens_decoder:
                return self.added_tokens_decoder[ids]
            else:
                return self._convert_id_to_token(ids)
        tokens = []
        for index in ids:
            if skip_special_tokens and index in self.all_special_ids:
                continue
            if index in end_flags:
                tokens.append('.')
                break
            if index in self.added_tokens_decoder:
                tokens.append(self.added_tokens_decoder[index])
            else:
                tokens.append(self._convert_id_to_token(index))
        return tokens






