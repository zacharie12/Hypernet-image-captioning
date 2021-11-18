import torch
from torch.utils.data import DataLoader
from data_loader import get_dataset, get_styled_dataset, ConcatDataset
import pickle
from models.decoderlstm import DecoderGRU, DecoderRNN, AttentionGru
from utils import cap_to_text, cap_to_text_gt
from hypernet_attention import HyperNet
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
                    p('Predicted Humorous')
                with td():
                    p('GT Humorous')
                with td():
                    p('Predicted Romantic')
                with td():
                    p('GT Romantic')
        
            for (img_path, fac_pred, fac_gt, hum_pred, hum_gt, rom_pred, rom_gt) in output:
                caps = [fac_pred, fac_gt, hum_pred, hum_gt, rom_pred, rom_gt]
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
    model_path = '/cortex/users/algiser/checkpoint_hn/pretrain/epoch=70-step=6160.ckpt'
    num_ims = 50
    save_path = '/home/lab/cohenza4/www/StyleFlicker7k.html'

    # data
    with open("data/vocab.pkl", 'rb') as f:
        vocab = pickle.load(f)

    orig_dataset = get_dataset(img_path, cap_path, vocab, get_path=True)
    humor_dataset = get_styled_dataset(cap_path_humor, vocab)
    romantic_dataset = get_styled_dataset(cap_path_romantic, vocab)

    data_concat = ConcatDataset(orig_dataset, humor_dataset, romantic_dataset)
    print('Loading Model')
    model = HyperNet.load_from_checkpoint(model_path,feature_size=100, embed_size=100, hidden_size=150, vocab_size=len(vocab), vocab=vocab, strict=False)
    model = model.to('cpu')
    """
    model = HyperNet(embed_size=200, hidden_size=150, num_layers=2, type='gru', vocab_size=len(vocab), vocab=vocab)
    model = model.load_state_dict(state_dict=torch.load(model_path))
    model = model.to('cpu')
    """
    print('Finished Loading Model')
    model.eval()
    # get captioner
    styles = ["factual", "humorous", "romantic"]
    captioners = []
    for style in styles:
        style = torch.tensor([vocab(style)])
        style = style.type(torch.LongTensor)
        style_emb = model.captioner.embed(style)
        captioner = model.forward(style_emb)
        
        captioner_copy = AttentionGru(2048, captioner.feature_out, 
                                    captioner.embedding_dim,
                                    captioner.hidden_dim,
                                    captioner.vocab_size,
                                    num_layers=captioner.num_layers
                                    )
        
        captioner_copy.load_state_dict(copy.deepcopy(captioner.state_dict()))
        captioner_copy.eval()
        captioners.append(captioner_copy)
        #captioners.append(captioner)

    i = 0
    output = []
    old_cap = torch.zeros((0,0))
    for ((img_tensor, fac_cap, img_name), hum_cap, rom_cap) in data_concat:
        line = []
        if i == num_ims:
            break
        line.append(img_name)
        img_feats = model.image_encoder(torch.unsqueeze(img_tensor, 0).float())
        gt_caps = [fac_cap, hum_cap, rom_cap]
        for k, captioner in enumerate(captioners):
            cap , _= captioner.infer(img_feats, vocab.w2i['<s>'])
            cap = torch.squeeze(cap)
            line.append(cap_to_text(cap, vocab))
            line.append(cap_to_text_gt(gt_caps[k], vocab))

        output.append(line)
        i += 1
        old_cap = hum_cap

    build_html(output, save_path)
