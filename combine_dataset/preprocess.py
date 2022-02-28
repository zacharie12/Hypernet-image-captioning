from collections import Counter
import tldextract
import numpy as np



def main():

    '''
    cap_in = "data/romantic/romantic_train.txt"
    cap_train = "data/rom_cap_train.txt"
    cap_test = "data/rom_cap_test.txt"
    cnt = 0
    with open(cap_in, 'r') as f:
        lines = f.readlines()
        print(len(lines))
        text_file = open(cap_train, "a+")
        for line in lines:
            if cnt == 6000:
                text_file.close()        
                text_file = open(cap_test, "a+")
            text_file.write(line)
            cnt +=1

    '''


    
    caption_CC_train = 'data/train_cap_100.txt'
    caption_CC_val = 'data/val_cap_100.txt'
    caption_CC_test = 'data/test_cap_100.txt'
    caption_fac = 'data/fac_cap_train.txt'
    caption_hum = 'data/humor_cap_train.txt'
    caption_rom = 'data/rom_cap_train.txt'
    caption_fac_test = 'data/fac_cap_test.txt'
    caption_hum_test = 'data/humor_cap_test.txt'
    caption_rom_test = 'data/rom_cap_test.txt'
    zero_shot_captions = 'data/one_shot_captions.txt'
    out_dir = "data/all_caption.txt"
    cnt = 0
    list_dir = [caption_CC_val, zero_shot_captions]

    with open(caption_CC_train, 'r') as f:
        lines = f.readlines()
        print(len(lines))
        text_file = open(out_dir, "a+")
        for line in lines:
            text_file.write(line)
            cnt += 1
        f.close()

    for dir in list_dir:
        with open(dir, 'r') as f:
            lines = f.readlines()
            print(len(lines))
            for line in lines:
                text_file.write(line)
                cnt += 1
            f.close()

    with open(caption_fac, 'r') as f:
            lines = f.readlines()
            print(len(lines))
            for line in lines:
                line = line.replace("        ", '     ')
                tmp = line.split('     ')
                if len(tmp) == 2:
                    text_file.write("{}     factual\n".format(line.replace("\n", '')))
                else:
                    line = line.replace("\t", '     ')
                    text_file.write("{}     factual\n".format(line.replace("\n", '')))
                cnt += 1
            f.close()


    with open(caption_hum, 'r') as f:
            lines = f.readlines()
            print(len(lines))
            for line in lines:
                text_file.write("{}     {}     humour\n".format(cnt, line.replace("\n", '')))
                cnt += 1
            f.close()

    with open(caption_rom, 'r') as f:
            lines = f.readlines()
            print(len(lines))
            for line in lines:
                text_file.write("{}     {}     romantic\n".format(cnt, line.replace("\n", '')))
                cnt += 1
            f.close()

    print("cnt = ", cnt)

 
if __name__ == '__main__':
    main()