import torch
from torch.utils.data import DataLoader
from cc_dataloader import ConceptualCaptions, collate_fn_visualize, get_dataset
import pickle
from models.decoderlstm import  AttentionGru
from build_vocab import Vocab
import dominate
from dominate.tags import *
import copy 
from cc_train_gru import Gru
from cc_train_hypernet import HyperNetCC
from utils import cap_to_text, cap_to_text_gt, get_domain_list

def build_html(output, out_path):
    doc = dominate.document(title='CC')
    with doc:
        with table().add(tbody()):
            with tr():
                with td():
                    p('Image')
                with td():
                    p('Predicted')
                with td():
                    p('GT')
       
            for (img_path, pred, gt) in output:
                caps = [pred, gt]
                with tr():
                    with td():
                        img(src=img_path)
                    for cap in caps:
                        with td():
                            p(cap)

    with open(out_path, 'w') as outfile:
            outfile.write(str(doc))

if __name__ == "__main__":
    img_path = 'data/200_conceptual_images_val/'
    img_path_train = 'data/200_conceptual_images_train/'
    cap_path = 'data/test_cap_100.txt'
    glove_path = "/cortex/users/cohenza4/glove.6B.100d.txt"
    model_path = "/cortex/users/cohenza4/checkpoint/HN/histograme_log/epoch=32-step=5378.ckpt"
    save_path = '/home/lab/cohenza4/www/CC_hyper.html'
    cap_dir_train = 'data/train_cap_100.txt'
    
    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    num_ims = 200
    # data
    with open("data/vocab_CC.pkl", 'rb') as f:
        vocab = pickle.load(f)

    test_data = get_dataset(img_path, cap_path, vocab, get_path=True)
    test_loader = DataLoader(test_data, batch_size=1, num_workers=2,
                            shuffle=False, collate_fn= collate_fn_visualize)

    print('Loading Model')
    domain_emb = 'histograme log'
    list_domain = get_domain_list(cap_dir_train, cap_path)  
    model = HyperNetCC(200, 200, 200, len(vocab), vocab, list_domain, 0.001, False, 0.3, 10, domain_emb) 
    model = model.load_from_checkpoint(checkpoint_path=model_path, vocab=vocab, list_domain=list_domain, embedding=domain_emb)

    i = 0
    output = []
    
     
    for imgs, caps, lengths, domains, img_name in test_loader:
        line = []
        if i == num_ims:
            break

        line.append(img_name[0])
        domain = domains[0]
        if model.embedding == "embedding":
            domain = torch.tensor(model.dict_domain[domain])
            domain = domain.type(torch.LongTensor)
            style_embed = model.embed(domain)
        elif model.embedding == 'one hot':
            domain = model.dict_domain[domain]
            style_embed = torch.tensor(model.embed[domain])
            style_embed = style_embed.type(torch.FloatTensor)
        else:
            domain = model.dict_domain[domain]
            domain =  torch.tensor(domain)
            domain = domain.type(torch.FloatTensor).to(model.device)
            style_embed = model.embed(domain)
        captioner = model.hypernet.forward(style_embed)
        img_feats = model.hypernet.image_encoder(imgs.float())
        img_feats = model.image_encoder(imgs.float()) 
        cap , _= captioner(img_feats, caps.long(), 0.0)
        cap = torch.squeeze(cap)
        line.append(cap_to_text(cap, vocab))
        line.append(cap_to_text_gt(caps[0], vocab))   
        output.append(line)
        i += 1
    build_html(output, save_path)
