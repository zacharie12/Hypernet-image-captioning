from collections import Counter
import tldextract
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import glob
from pathlib import Path
import os
from os import listdir
import PIL
from PIL import Image
def get_count(p, p2):
    with open(p, 'r') as f:
        lines = f.readlines()
        domain_count = Counter()
        for line in lines:
            url = line.split('\t')[1]
            domain = tldextract.extract(url)[1]  # get domain
            domain_count[domain] += 1
    with open(p, 'r') as f:
        lines = f.readlines()
        for line in lines:
            url = line.split('\t')[1]
            domain = tldextract.extract(url)[1]  # get domain
            domain_count[domain] += 1
    return domain_count


def main():

    '''
    
    file_path_train = '/cortex/data/images/conceptual_captions/Train_GCC-training.tsv'
    file_path_val = '/cortex/data/images/conceptual_captions/Validation_GCC-1.1.0-Validation.tsv'
    count_train = get_count(file_path_train, file_path_val)
    sorted_domain = count_train.most_common()
    for i in range(0, 100, 10):
        print(i, ": ", sorted_domain[i][1])
    print("200: ",  sorted_domain[100][1] )



    

    

    top_k_domain_val = sorted_domain_val[:100]
    domain_list_val_name, domain_list_val_num = [], []
    for j in range(len(top_k_domain_val)):
        domain_list_val_name.append(top_k_domain_val[j][0])
        domain_list_val_num.append(top_k_domain_val[j][1])
    #print("val name", domain_list_val_name)
    #print("train num  ", domain_list_train_num)
    #print("val num  ", domain_list_val_num)
    sorted_train = sorted(domain_list_train_name)  
    sorted_val= sorted(domain_list_val_name)  
    for i in range(len(sorted_train)):
        if sorted_train[i] != sorted_val[i]:
            print("train ", sorted_train[i])
            print("val ", sorted_val[i])


      with open(out_cap, "a+") as out_file:
        with open(out_caption1, 'r') as f:
            lines = f.readlines()
            for line in lines:
                x = line.split("     ")
                ids.append(x[0])
                captions.append(x[1])
                domain = x[2].replace("\n", '')
                domains.append(domain)
                out_file.write(line)
            counter = Counter(domains)
            print(counter.keys())
            print(counter.values())

        with open(out_caption2, 'r') as f:  
            lines = f.readlines()
            for line in lines:
                x = line.split("     ")
                domain = x[2].replace("\n", '')
                if domain not in counter.keys():
                    out_file.write(line)  

    out_file.close()       

    '''

    image_train_in = '/dsi/shared/datasets/howto100m/hn_caption/200_conceptual_images_train'
    image_val_test_in = '/dsi/shared/datasets/howto100m/hn_caption/200_conceptual_images_val'
    cap_train_in = 'data/train_cap_100.txt'
    cap_train_out = '/dsi/shared/datasets/howto100m/hn_caption/train_cap'
    cap_val_out = '/dsi/shared/datasets/howto100m/hn_caption/val_cap'
    cap_test_out = '/dsi/shared/datasets/howto100m/hn_caption/test_cap'
    cap_val_in = 'data/val_cap_100.txt'
    cap_test_in = 'data/test_cap_100.txt'
    out_img = '/dsi/shared/datasets/howto100m/hn_caption/cc_test_img/'
    cnt = 0
    with open(cap_test_out, 'r') as f: 
        lines = f.readlines()
        for line in lines:
            x = line.split("     ")
            ids = x[0]
            for images in os.listdir(image_val_test_in):
                if ids == images:
                    path_in = image_val_test_in + '/' + images
                    path_out = out_img + images
                    im = Image.open(path_in)
                    im.save(path_out)
                    cnt += 1
                    break

        print("count = ", cnt)

    '''
    imgFiles = list(Path(image_val_test_in).glob('*.jpg'))
    imgIds = np.array([int(f.stem.split()[0]) for f in imgFiles])
    num_cap_val,  num_cap_test = 0, 0 
    with open(cap_val_out, 'r') as cap_val:
        lines = cap_val.readlines()
        for line in lines:
            x = line.split("     ")
            ids = x[0]
            for id in imgIds:
                id_img = str(id)+ '.jpg'
                if ids == id_img:


    '''
            



        

   

    '''    
    out_cap = 'cc_big_100.txt'
    out_img = 'cc_big_100'
    ids, captions, domains = [], [], []

    

    '''         
      

if __name__ == '__main__':
    main()








   
