from collections import Counter
import tldextract


def get_count(p):
    with open(p, 'r') as f:
        lines = f.readlines()
        domain_count = Counter()
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
    file_path = '/cortex/data/images/conceptual_captions/Train_GCC-training.tsv'
    out_path = 'data/conceptual_domains.tsv'
    xmin = 50
    xmax = 500
    print(f'Filtering domains with less than {xmin} and more than {xmax} examples')
    count = get_count(file_path)
    count = filter_count(count, xmin, xmax)
    print(f'Number of examples: {sum(count.values())}')
    domain_list = count.keys()
    print(f'Number of domains: {len(domain_list)}')
    output = []
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            url = line.split('\t')[1]
            domain = tldextract.extract(url)[1]  # get domain
            if domain in domain_list:
                output.append(f'{line[:-1]}\t{domain}')
    print('Writing output to file')
    with open(out_path, 'w') as f:
        for line in output:
            f.write(line)

    print('Done!')



if __name__ == '__main__':
    main()