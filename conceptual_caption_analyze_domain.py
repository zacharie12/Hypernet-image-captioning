from collections import Counter
import tldextract
from matplotlib import pyplot as plt
import numpy as np

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



    '''

    '''

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


    '''
    
    file = 'data/200_conceptual_val.txt'
    ids, captions, domains = [], [], []
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            x = line.split("     ")
            ids.append(x[0])
            captions.append(x[1])
            domain = x[2].replace("\n", '')
            domains.append(domain)
        counter = Counter(domains)
        print(counter.values())
        print(len(counter))
    

    '''
    file1 = 'data/conceptual_val.txt'
    file2 = 'data/CC_val.txt'
    ids, captions, domains = [], [], []
    with open(file1, 'r') as f:
        lines = f.readlines()
    with open(file2, "w") as g:
        for line in lines:
            x = line.split("     ")
            domain = x[2]
            if domain != "akamaihd\n" and domain != "rackcdn\n" and domain != "photobucket\n" and  domain != "ibtimes\n" and domain != "nasa\n" and domain != "els-cdn\n":
                g.write(line)
                

    '''
 
           

if __name__ == '__main__':
    main()








   
