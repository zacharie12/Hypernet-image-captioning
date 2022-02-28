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
    out_image = 'data/one_shot_images/'
    out_caption = 'data/one_shot_captions.txt'
    images = []
    count = get_count(file_path_train, file_path_val)
    sorted_domain = count.most_common()
    top_k_domain = sorted_domain[189:]
    domain_list_name, domain_list_num= [], []
    for j in range(len(top_k_domain)):
        domain_list_name.append(top_k_domain[j][0])
    output_caption_domain = []
    idx, bad_images,index, tmp = 1, 0, 0, 10
    bad_domain, i = 0, -1
    is_bad_domain = False
    img_dir = out_image

    while (i - bad_domain) < 20:
        if is_bad_domain:
            bad_domain += 1
            is_bad_domain = False
            can_save = False
        else:
            can_save = True

        i += 1
        print("curr domain is",  domain_list_name[i])
        if i > 0:
            domain_list_num.append(cnt)
            print("i = ", i , "  domain done = ", (i - bad_domain), "image in this domain = ", cnt, "image in val ", max(0, (cnt-50)))

            if ((i - bad_domain) % 10 == 0) and can_save:
                for ca in range(tmp):
                    text_file = open(out_caption, "a+")
                    for j in range(domain_list_num[ca]):
                        text_file.write("{}     {}     {}\n".format(output_caption_domain[index][0], output_caption_domain[index][1], output_caption_domain[index][2]))
                        index += 1
                    text_file.close()
                output_caption_domain, domain_list_num, index = [], [], 0
                  
        cnt, cnt_badimg_domain = 0, 0
        with open(file_path_train, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if (cnt_badimg_domain > 10 and cnt == 0) or  ((cnt_badimg_domain - cnt) > 500):
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
                        images.append([image, img_dir, idx])
                        image.save("{}{}.jpg".format(img_dir, idx))
                        cap_domain = ["{}.jpg".format(idx), captions, domain]
                        output_caption_domain.append(cap_domain)
                        cnt += 1
                        
                        if cnt == 50:
                            break
                    except PIL.UnidentifiedImageError:
                        bad_images +=1
                        cnt_badimg_domain += 1
                    except ConnectionError:
                        bad_images +=1
                        cnt_badimg_domain += 1
                    except OSError:
                        bad_images +=1
                        cnt_badimg_domain += 1
                    except (http_incompleteRead, urllib3_incompleteRead , ConnectionError, ValueError, ProtocolError, RemoteDisconnected):
                        bad_images +=1
                        cnt_badimg_domain += 1

                        

        with open(file_path_val, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if cnt ==  50:
                    break
                if (cnt_badimg_domain > 20 and cnt == 0) or  ((cnt_badimg_domain - cnt) > 50):
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
                        images.append([image, img_dir, idx])
                        image.save("{}{}.jpg".format(img_dir, idx))
                        cap_domain = ["{}.jpg".format(idx), captions, domain]
                        output_caption_domain.append(cap_domain)
                        cnt += 1
                        if cnt ==  50:
                            break
                        if (cnt_badimg_domain > 20 and cnt == 0) or  ((cnt_badimg_domain - cnt) > 50):
                            is_bad_domain = True
                            break
                    except PIL.UnidentifiedImageError:
                        bad_images +=1
                        cnt_badimg_domain += 1
                    except ConnectionError:
                        bad_images +=1
                        cnt_badimg_domain += 1
                    except OSError:
                        bad_images +=1
                        cnt_badimg_domain += 1
                    except (http_incompleteRead, urllib3_incompleteRead , ConnectionError, ValueError, ProtocolError, RemoteDisconnected):
                        bad_images +=1
                        cnt_badimg_domain += 1
    
    #domain_list_num.append(cnt)
    
    #print("i: ", i)
    '''
    for im in range(len(images)):
        image, img_dir, im_idx = images[im][0], images[im][2], images[im][2]
        try:
            image.save("{}{}.jpg".format(img_dir, im_idx))
        except OSError:
            Image.open(image).convert('RGB').save("{}{}.jpg".format(img_dir, im_idx))
        
    '''
    print('Done!')
    print("bad_domain = ", bad_domain)
    print("idx", idx)
    print("index ", index)


if __name__ == '__main__':
    main()