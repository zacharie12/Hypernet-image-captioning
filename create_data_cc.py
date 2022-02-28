from collections import Counter
import tldextract
from PIL import Image
import skimage.transform
import requests
import PIL
import cv2
from matplotlib import cm
from torchvision.utils import save_image
import numpy as np
from torchvision import transforms
import torch 
from requests.exceptions import ConnectionError
import os
from http.client import IncompleteRead as http_incompleteRead
from urllib3.exceptions import IncompleteRead as urllib3_incompleteRead
from requests.exceptions import ConnectionError
from http.client import RemoteDisconnected
from urllib3.exceptions import ProtocolError


def get_count(p1, p2):
    with open(p1, 'r') as f:
        lines = f.readlines()
        domain_count = Counter()
        for line in lines:
            url = line.split('\t')[1]
            domain = tldextract.extract(url)[1]  # get domain
            domain_count[domain] += 1
    with open(p2, 'r') as f:
        lines = f.readlines()
        for line in lines:
            url = line.split('\t')[1]
            domain = tldextract.extract(url)[1]  # get domain
            domain_count[domain] += 1
    return domain_count


def filter_count(count, xmin, xmax):
    for key in list(count.keys()):
        if not (count[key] <= xmax and count[key] >= xmin):
            del count[key]
    return count


def main():
    file_path_train = '/cortex/data/images/conceptual_captions/Train_GCC-training.tsv'
    file_path_val = '/cortex/data/images/conceptual_captions/Validation_GCC-1.1.0-Validation.tsv'
    out_image = 'data/cc_big/'
    out_caption = 'data/cc_big.txt'
    num_domain, max_im, cnt_bad_im, cnt_img = 5, 300, 0, 0
    count = get_count(file_path_train, file_path_val)
    sorted_domain = count.most_common()
    top_k_domain = sorted_domain[28:]
    domain_list_name, domain_list_num= [], []
    for j in range(len(top_k_domain)):
        domain_list_name.append(top_k_domain[j][0])
    idx= 1320000
    bad_domain, i = 0, -1
    is_bad_domain = False
    cnt_bad_im, cnt_img = 0, 0  
    while (i - bad_domain) < num_domain:
        if is_bad_domain:
            bad_domain += 1
            cnt_bad_im = 0
        i += 1
        print("i = ", i , "  domain done = ", (i - bad_domain), "image in this domain = ", cnt_img)
        print("curr domain is",  domain_list_name[i])
        with open(file_path_train, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if (cnt_bad_im > 50 and cnt_img == 0) or  ((cnt_bad_im - cnt_img) > 500):
                    is_bad_domain = True
                    break
                url = line.split('\t')[1]
                captions = line.split('\t')[0]
                domain = tldextract.extract(url)[1]  # get domain
                if domain == domain_list_name[i]:
                    try:
                        size_urls = len(url)
                        search_url = url[:size_urls - 1]
                        image = Image.open(requests.get(search_url, stream=True).raw)
                        idx += 1
                        cnt_img += 1
                        image.save("{}{}.jpg".format(out_image, idx))
                        text_file = open(out_caption, "a+")
                        text_file.write("{}     {}     {}\n".format("{}.jpg".format(idx), captions, domain))
                        text_file.close()

                        if cnt_img == max_im:
                            break
                    except PIL.UnidentifiedImageError:
                        cnt_bad_im += 1
                    except ConnectionError:
                        cnt_bad_im += 1
                    except OSError:
                        cnt_bad_im += 1
                    except (http_incompleteRead, urllib3_incompleteRead , ConnectionError, ValueError, ProtocolError, RemoteDisconnected):
                        cnt_bad_im += 1        

        with open(file_path_val, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if cnt_img == max_im:
                    break
                if (cnt_bad_im > 50 and cnt_img == 0) or  ((cnt_bad_im - cnt_img) > 500):
                    is_bad_domain = True
                    break
                url = line.split('\t')[1]
                captions = line.split('\t')[0]
                domain = tldextract.extract(url)[1]  # get domain
                if domain == domain_list_name[i]:
                    try:
                        size_urls = len(url)
                        search_url = url[:size_urls - 1]
                        image = Image.open(requests.get(search_url, stream=True).raw)
                        idx += 1
                        cnt_img += 1
                        image.save("{}{}.jpg".format(out_image, idx))
                        text_file.write("{}     {}     {}\n".format("{}.jpg".format(idx), captions, domain))
                        if cnt_img == max_im:
                            break
                        if (cnt_bad_im > 50 and cnt_img == 0) or  ((cnt_bad_im - cnt_img) > 500):
                            is_bad_domain = True
                            break
                    except PIL.UnidentifiedImageError:
                        cnt_bad_im += 1
                    except ConnectionError:
                        cnt_bad_im += 1
                    except OSError:
                        cnt_bad_im += 1
                    except (http_incompleteRead, urllib3_incompleteRead , ConnectionError, ValueError, ProtocolError, RemoteDisconnected):
                        cnt_bad_im += 1

                
        
    
    print('Done!')
    print("bad_domain = ", bad_domain)
    print("idx", idx)



if __name__ == '__main__':
    main()