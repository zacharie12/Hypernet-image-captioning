import torch
from torch.utils.data import DataLoader
from data_loader import get_dataset, get_styled_dataset, ConcatDataset, flickr_collate_fn_visualize
import pickle
from models.decoderlstm import  AttentionGru
from train_attention_gru import CaptionAttentionGru
from utils import cap_to_text, cap_to_text_gt, cap_to_text_gt_viz
from build_vocab import Vocab
import dominate
from dominate.tags import *
import copy 

def build_html(output, out_path):
    doc = dominate.document(title='StyleFlicker7k')
    with doc:
        with table().add(tbody()):
            with tr():
                with td():
                    p('Image')
                with td():
                    p('Predicted Factual')
                with td():
                    p('GT Factual')
                with td():
                    p('GT Humorous')
                with td():
                    p('GT Romantic')
        
            for (img_path, fac_pred, fac_gt, hum_gt, rom_gt) in output:
                caps = [fac_pred, fac_gt, hum_gt, rom_gt]
                with tr():
                    with td():
                        img(src=img_path)
                    for cap in caps:
                        with td():
                            p(cap)

    with open(out_path, 'w') as outfile:
            outfile.write(str(doc))

if __name__ == "__main__":
    img_path = "data/flickr7k_images"
    cap_path = "data/factual_train.txt"
    cap_path_humor = "data/humor/funny_train.txt"
    cap_path_romantic = "data/romantic/romantic_train.txt"
    glove_path = "/cortex/users/cohenza4/glove.6B.100d.txt"
    model_path = '/cortex/users/cohenza4/checkpoint_gru/factual/epoch=11-step=484.ckpt'
    num_ims = 50
    save_path = '/home/lab/cohenza4/www/StyleFlicker7k_gru_factual.html'

    # data
    with open("data/vocab.pkl", 'rb') as f:
        vocab = pickle.load(f)

    orig_dataset = get_dataset(img_path, cap_path, vocab, get_path=True)
    humor_dataset = get_styled_dataset(cap_path_humor, vocab)
    romantic_dataset = get_styled_dataset(cap_path_romantic, vocab)

    data_concat = ConcatDataset(orig_dataset, humor_dataset, romantic_dataset)
    lengths = [int(len(data_concat)*0.8),
               len(data_concat) - int(len(data_concat)*0.8)]
    train_data, val_data = torch.utils.data.random_split(data_concat, lengths)

    train_loader = DataLoader(train_data, batch_size=1, num_workers=1,
                              shuffle=False, collate_fn=flickr_collate_fn_visualize)
    val_loader = DataLoader(val_data, batch_size=1, num_workers=1,
                            shuffle=False, collate_fn=flickr_collate_fn_visualize)

    print('Loading Model')
    model = CaptionAttentionGru(256, 100, 200, len(vocab), vocab, 1, lr=0.0003)
    model = model.load_from_checkpoint(model_path, vocab=vocab)

    i = 0
    output = []
     
    for ((img_tensor, fac_cap, img_name), (hum_cap, lengths), (rom_cap, lengths)) in val_loader:
        line = []
        if i == num_ims:
            break

        line.append(img_name[0])
        img_feats = model.image_encoder(img_tensor.float())
        
        gt_caps = [fac_cap, hum_cap, rom_cap]
        
        #cap , _= model.captioner.infer(img_feats, vocab.w2i['<s>'])
        cap , _= model.captioner(img_feats, fac_cap[0].long(), 1.0)
        '''
        cap = model.captioner.infer(img_feats, vocab.w2i['</s>'], 40)
        line.append(cap)
        '''
        #cap = torch.Tensor(cap)
        cap = torch.squeeze(cap)
        line.append(cap_to_text(cap, vocab))
        fact_gt =True
        for gt_cap in gt_caps:
            if fact_gt:
                line.append(cap_to_text_gt_viz(gt_cap[0], vocab))
            else:
                line.append(cap_to_text_gt(gt_cap[0], vocab))   
            fact_gt = False
        output.append(line)
        i += 1


    build_html(output, save_path)



    """
    for ((img_tensor, fac_cap, img_name), hum_cap, rom_cap) in data_concat:
        line = []
        if i == num_ims:
            break
        line.append(img_name)
        img_feats = model.image_encoder(torch.unsqueeze(img_tensor, 0).float())

        gt_caps = [fac_cap, hum_cap, rom_cap]

        #cap , _= model.captioner.infer(img_feats, vocab.w2i['<s>'])
        cap , _= model.captioner(img_feats, fac_cap.long().unsqueeze(0), 1.0)
        '''
        cap = model.captioner.infer(img_feats, vocab.w2i['</s>'], 40)
        line.append(cap)
        '''
        #cap = torch.Tensor(cap)
        cap = torch.squeeze(cap)
        line.append(cap_to_text(cap, vocab))

        for gt_cap in gt_caps:
            line.append(cap_to_text_gt(gt_cap, vocab))

        output.append(line)
        i += 1


    build_html(output, save_path)
    """ 