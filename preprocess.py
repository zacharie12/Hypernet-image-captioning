import re
import shutil
import os
import random


def select_7k_images(c_type='humor'):
    '''30k -> 7k'''
    # open data/type/train.p
    with open('data/' + c_type + '/train.p', 'r') as f:
        res = f.readlines()

    # extract img names
    img_list = []
    r = re.compile("\d*_")
    for line in res:
        if len(line) < 10:
            continue
        line = r.search(line)
        line = line.group(0)[:-1] + '.jpg'
        img_list.append(line)

    # copy imgs
    for img_name in img_list:
        shutil.copyfile('data/flickr30k_images/' + img_name,
                        'data/flickr7k_images/' + img_name)


def select_factual_captions(order_like='humor'):
    '''30k -> 7k'''
    img_list = []
    if order_like is not None:
        with open('data/' + order_like + '/train.p', 'r') as f:
            res = f.readlines()

        # extract img names
        img_list = []
        r = re.compile("\d*_")
        for line in res:
            if len(line) < 10:
                continue
            line = r.search(line)
            line = line.group(0)[:-1] + '.jpg'
            img_list.append(line)

    # get filenames in flickr7k_images
    filenames = os.listdir('data/flickr7k_images/')
    # open data/results_20130124.token
    with open('data/results_20130124.token', 'r') as f:
        res = f.readlines()

    if order_like is None:
        # write out
        with open('data/factual_train.txt', 'w') as f:
            r = re.compile('\d*.jpg')
            for line in res:
                img = r.search(line)
                img = img.group(0)
                if img in filenames:
                    f.write(line)
    else:
        with open('data/factual_train.txt', 'w') as f:
            r = re.compile('\d*.jpg')
            for i, img_name in enumerate(img_list):
                for line in res:
                    img = r.search(line)
                    img = img.group(0)
                    if img == img_name:
                        f.write(line)

        
        


def random_select_test_images(num=100):
    '''select test images randomly'''
    # get filenames in flickr7k, 30k_images
    filenames_7k = os.listdir('data/flickr7k_images/')
    filenames_30k = os.listdir('data/flickr30k_images')

    filenames = list(set(filenames_30k) - set(filenames_7k))
    print("img_num: " + str(len(filenames)))
    random.seed(24)
    selected = random.sample(filenames, num)

    # copy images
    for img_name in selected:
        shutil.copyfile('data/flickr30k_images/' + img_name,
                        'test_images/' + img_name)


if __name__ == '__main__':
    print('Selecting humor images')
    select_7k_images('humor')
    print('Selecting romantic images')
    select_7k_images('romantic')
    print('Selecting captions')
    select_factual_captions(order_like='humor')
    print('Selecting test images')
    random_select_test_images()