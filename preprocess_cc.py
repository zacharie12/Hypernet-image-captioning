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

    

    file_out_val = 'data/val_cap_100.txt'
    file_out_test = 'data/test_cap_100.txt'
    with open(file_out_val, 'r') as f:
            lines = f.readlines()
            print(len(lines))
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
        counter = Counter(dict(filter(lambda x: x[1] >= 21, counter.items())))
        list_domain = list(counter.keys())
        list_domain.remove("ebayimg")


    file_origine = 'data/200_conceptual_val.txt'
    file_out_val = 'data/val_cap_100.txt'
    file_out_test = 'data/test_cap_100.txt'
    

    ids, captions, domains = [], [], []
    for cur_domain in list_domain:
        cnt = 0
        with open(file_origine, 'r') as f:
            lines = f.readlines()
            with open(file_out_val, "a+") as g:
                for line in lines:
                    x = line.split("     ")
                    domain = x[2].replace("\n", '')
                    if domain == cur_domain and cnt < 10:
                        g.write(line)
                        cnt += 1
            g.close()
        f.close
        with open(file_origine, 'r') as f:
            lines = f.readlines()
            with open(file_out_test, "a+") as g:
                for line in lines:
                    x = line.split("     ")
                    domain = x[2].replace("\n", '')
                    if domain == cur_domain and cnt < 10:
                        cnt += 1
                    elif domain == cur_domain and cnt > 9 and cnt < 30:
                        g.write(line)
                        cnt += 1
                        
        '''
        
            
    
    
                      

if __name__ == '__main__':
    main()








   
