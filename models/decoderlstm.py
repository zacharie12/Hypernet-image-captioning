import torch
from torch import nn
from torch.nn import functional as F
import pickle
from utils import set_all_parameters, flip_parameters_to_tensors, WordVectorLoader, cap_to_text, cap_to_text_gt, sample_multinomial_topk, clean_sentence
from .attention import BahdanauAttention
import numpy as np
from models.encoder import EncoderCNN, EncoderLstm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AttentionGru(nn.Module):
    def __init__(self, num_features, feature_out, embedding_dim, hidden_dim, vocab_size, num_layers=1,  p=0.0):
        super(AttentionGru, self).__init__()
        self.num_features = num_features
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        # scale the inputs to softmax
        self.sample_temp = 0.5

        self.feature_fc = nn.Sequential(
            nn.Linear(num_features, feature_out),
            nn.ReLU(),
            nn.Linear(feature_out, feature_out)
        )
        # embedding layer that turns words into a vector of a specified size
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        # LSTM will have a single layer of size 512 (512 hidden units)
        # it will input concatinated context vector (produced by attention)
        # and corresponding hidden state of Decoder
        self.gru = nn.GRUCell(embedding_dim + feature_out, hidden_dim)
        self.layers = None
        if num_layers > 1:
            self.layers = nn.ModuleList([nn.GRUCell(input_size=hidden_dim, hidden_size=hidden_dim) for layer in range(num_layers-1)])

        # produce the final output
        self.fc = nn.Linear(hidden_dim, vocab_size)

        # add attention layer
        self.attention = BahdanauAttention(feature_out, hidden_dim)
        # dropout layer
        self.drop = nn.Dropout(p=p)
        # add initialization fully-connected layers
        # initialize hidden state and cell memory using average feature vector
        # Source: https://arxiv.org/pdf/1502.03044.pdf
        self.init_h = nn.Linear(feature_out, hidden_dim)

    def forward(self, features, captions, sample_prob = 0.0):
            """Arguments
            ----------
            - captions - image captions
            - features - features returned from Encoder
            - sample_prob - use it for scheduled sampling
            Returns
            ----------
            - outputs - output logits from t steps
            - atten_weights - weights from attention network
            """
            # create embeddings for captions of size (batch, sqe_len, embed_dim)
            features = self.feature_fc(features)
            embed = self.embed(captions)
            h = self.init_hidden(features)
            if self.layers:
                for layer in self.layers:
                    h = layer(h, h)
            seq_len = captions.size(1)
            feature_size = features.size(1)
            batch_size = features.size(0)
            sample_prob_tmp = sample_prob
            # these tensors will store the outputs from lstm cell and attention weights
            outputs = torch.zeros(batch_size, seq_len, self.vocab_size).to(device)
            atten_weights = torch.zeros(batch_size, seq_len, feature_size).to(device)

            # scheduled sampling for training
            # we do not use it at the first timestep (<start> word)
            # but later we check if the probability is bigger than random
            for t in range(seq_len):
                sample_prob = 0.0 if t == 0 else sample_prob_tmp
                use_sampling = np.random.random() < sample_prob
                if use_sampling == False:
                    if t == 0:
                        word_embed = embed[:,t,:]
                        word_embed[:][:][:][:][:][:] = 0
                    else:
                        #gt_idx = torch.squeeze(captions[t-1])
                        #ref_text = cap_to_text_gt(gt_idx, vocab, tokenized=False)
                        word_embed = embed[:,t-1,:]
                else:
                    # use sampling temperature to amplify the values before applying softmax
                    scaled_output = output / self.sample_temp
                    #caps_pred_idx = torch.squeeze(scaled_output)
                    #hyp_text = cap_to_text(caps_pred_idx, vocab, tokenized=False)
                    scoring = F.log_softmax(scaled_output, dim=1)
                    top_idx = scoring.topk(1)[1]
                    word_embed = self.embed(top_idx).squeeze(1)    
                context, atten_weight = self.attention(features, h)
                # input_concat shape at time step t = (batch, embedding_dim + hidden_dim)
                input_concat = torch.cat([word_embed, context], 1)
                h = self.gru(input_concat, h)
                if self.layers:
                    for layer in self.layers:
                        h = layer(h, h)
                h = self.drop(h)
                output = self.fc(h)

                outputs[:, t, :] = output
                atten_weights[:, t, :] = atten_weight
                '''
                with open("data/vocab.pkl", 'rb') as f:
                    vocab = pickle.load(f)
                gt_word_previous = captions[:, t-1]
                gt_word = captions[:, t]
                pred_word = torch.argmax(output, dim=1)
                st_gt_prev = cap_to_text_gt(gt_word_previous, vocab)
                st_gt = cap_to_text_gt(gt_word, vocab)
                st_pred = cap_to_text_gt(pred_word, vocab)
                a=1
                '''
            return outputs, atten_weights

    def init_hidden(self, features):

        """Initializes hidden state and cell memory using average feature vector.
        Arguments:
        ----------
        - features - features returned from Encoder
        Retruns:
        ----------
        - h0 - initial hidden state (short-term memory)
        - c0 - initial cell state (long-term memory)
        """
        mean_annotations = torch.mean(features, dim = 1)
        h0 = self.init_h(mean_annotations)
        return h0


    def greedy_search(self, features, end_sentence=2, max_sentence = 20):

        """Greedy search to sample top candidate from distribution.
        Arguments
        ----------
        - features - features returned from Encoder
        - max_sentence - max number of token per caption (default=20)
        Returns:
        ----------
        - sentence - list of tokens
        """

        sentence = []
        weights = []
        input_word = torch.tensor(0).unsqueeze(0)
        h = self.init_hidden(features)
        if self.layers:
            for layer in self.layers:
                h = layer(h, h)
        while True:
            embedded_word = self.embed(input_word)
            context, atten_weight = self.attention(features, h)
            # input_concat shape at time step t = (batch, embedding_dim + context size)
            input_concat = torch.cat([embedded_word, context],  dim = 1)
            h = self.gru(input_concat, h)
            if self.layers:
                for layer in self.layers:
                    h = layer(h, h)
            h = self.drop(h)
            output = self.fc(h)
            scoring = F.log_softmax(output, dim=1)
            top_idx = scoring[0].topk(1)[1]
            sentence.append(top_idx.item())
            weights.append(atten_weight)
            input_word = top_idx
            if (len(sentence) >= max_sentence or top_idx == end_sentence):
                break
        return sentence, weights


    def infer(self, features, end_sentence=2, max_len=40):
        with open("data/vocab.pkl", 'rb') as f:
            vocab = pickle.load(f)
        features = self.feature_fc(features)
        output, atten_weights = self.greedy_search(features, end_sentence, max_len) 
        sentence = clean_sentence(output, vocab)
        return sentence



class AttentionLstm(nn.Module):
    """Attributes:
    - embedding_dim - specified size of embeddings;
    - hidden_dim - the size of RNN layer (number of hidden states)
    - vocab_size - size of vocabulary
    - p - dropout probability
    """
    def __init__(self, num_features, embedding_dim, hidden_dim, vocab_size, p =0.5):
        super(AttentionLstm, self).__init__()

        self.num_features = num_features
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        # scale the inputs to softmax
        self.sample_temp = 0.5

        # embedding layer that turns words into a vector of a specified size
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        # LSTM will have a single layer of size 512 (512 hidden units)
        # it will input concatinated context vector (produced by attention)
        # and corresponding hidden state of Decoder
        self.lstm = nn.LSTMCell(embedding_dim + num_features, hidden_dim)
        # produce the final output
        self.fc = nn.Linear(hidden_dim, vocab_size)

        # add attention layer
        self.attention = BahdanauAttention(num_features, hidden_dim)
        # dropout layer
        self.drop = nn.Dropout(p=p)
        # add initialization fully-connected layers
        # initialize hidden state and cell memory using average feature vector
        # Source: https://arxiv.org/pdf/1502.03044.pdf
        self.init_h = nn.Linear(num_features, hidden_dim)
        self.init_c = nn.Linear(num_features, hidden_dim)

    def forward(self, captions, features, sample_prob = 1.0):
        embed = self.embeddings(captions.long())
        h, c = self.init_hidden(features)
        seq_len = captions.size(1)
        feature_size = features.size(1)
        batch_size = features.size(0)
        outputs = torch.zeros(batch_size, seq_len, self.vocab_size).to(device)
        atten_weights = torch.zeros(batch_size, seq_len, feature_size).to(device)
        sample_prob_tmp = sample_prob
        for t in range(seq_len):
            sample_prob = 0.0 if t == 0 else sample_prob_tmp
            use_sampling = np.random.random() < sample_prob
            if use_sampling == False:
                if t == 0:
                    word_embed = embed[:,t,:]
                    word_embed[:][:][:][:][:][:] = 0
                else:
                    word_embed = embed[:,t-1,:]
            context, atten_weight = self.attention(features, h)
            input_concat = torch.cat([word_embed, context], 1)
            h, c = self.lstm(input_concat, (h,c))
            h = self.drop(h)
            output = self.fc(h)
            if use_sampling == True:
                scaled_output = output / self.sample_temp
                scoring = F.log_softmax(scaled_output, dim=1)
                top_idx = scoring.topk(1)[1]
                word_embed = self.embeddings(top_idx).squeeze(1)
            outputs[:, t, :] = output
            atten_weights[:, t, :] = atten_weight
        return outputs, atten_weights

    def init_hidden(self, features):

        mean_annotations = torch.mean(features, dim = 1)
        h0 = self.init_h( mean_annotations)
        c0 = self.init_c(mean_annotations)
        return h0, c0


    def greedy_search(self, features, end_sentence=2, max_sentence=30):

        sentence = []
        weights = []
        h, c= self.init_hidden(features)
        input_word = torch.tensor(0).unsqueeze(0)
        while True:
            embedded_word = self.embeddings(input_word)
            context, atten_weight = self.attention(features, h)
            # input_concat shape at time step t = (batch, embedding_dim + context size)
            input_concat = torch.cat([embedded_word, context],  dim = 1)
            h, c = self.lstm(input_concat, (h,c))
            h = self.drop(h)
            output = self.fc(h)
            scoring = F.log_softmax(output, dim=1)
            top_idx = scoring[0].topk(1)[1]
            sentence.append(top_idx.item())
            weights.append(atten_weight)
            input_word = top_idx
            if (len(sentence) >= max_sentence or top_idx == end_sentence):
                break
        return sentence, weights
    
    def infer(self, img, end_sentence=2, max_len=40):
        with open("data/vocab.pkl", 'rb') as f:
            vocab = pickle.load(f)
        features =  self.image_encoder(img)
        output, atten_weights = self.greedy_search(features, end_sentence, max_len) 
        sentence = clean_sentence(output, vocab)
        return sentence


class DecoderLstm(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        ''' Initialize the layers of this model.'''
        super().__init__()
    
        # Keep track of hidden_size for initialization of hidden state
        self.hidden_size = hidden_size
        
        # Embedding layer that turns words into a vector of a specified size
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        
        # The LSTM takes embedded word vectors (of a specified size) as input
        # and outputs hidden states of size hidden_dim
        self.lstm = nn.LSTM(input_size=embed_size, \
                            hidden_size=hidden_size, # LSTM hidden units 
                            num_layers=1, # number of LSTM layer
                            bias=True, # use bias weights b_ih and b_hh
                            batch_first=True,  # input & output will have batch size as 1st dimension
                            dropout=0, # Not applying dropout 
                            bidirectional=False, # unidirectional LSTM
                           )
        
        # The linear layer that maps the hidden state output dimension
        # to the number of words we want as output, vocab_size
        self.linear = nn.Linear(hidden_size, vocab_size)                     

        # initialize the hidden state
        # self.hidden = self.init_hidden()
        
    def init_hidden(self, batch_size):
        """ At the start of training, we need to initialize a hidden state;
        there will be none because the hidden state is formed based on previously seen data.
        So, this function defines a hidden state with all zeroes
        The axes semantics are (num_layers, batch_size, hidden_dim)
        """
        return (torch.zeros((1, batch_size, self.hidden_size), device=device), \
                torch.zeros((1, batch_size, self.hidden_size), device=device))

    def forward(self, features, captions):
        """ Define the feedforward behavior of the model """
        
        # Discard the <end> word to avoid predicting when <end> is the input of the RNN
        captions = captions[:, :-1]     
        
        # Initialize the hidden state
        batch_size = features.shape[0] # features is of shape (batch_size, embed_size)
        self.hidden = self.init_hidden(self.batch_size) 
                
        # Create embedded word vectors for each word in the captions
        embeddings = self.word_embeddings(captions) # embeddings new shape : (batch_size, captions length - 1, embed_size)
        
        # Stack the features and captions
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1) # embeddings new shape : (batch_size, caption length, embed_size)
        
        # Get the output and hidden state by passing the lstm over our word embeddings
        # the lstm takes in our embeddings and hidden state
        lstm_out, self.hidden = self.lstm(embeddings, self.hidden) # lstm_out shape : (batch_size, caption length, hidden_size)

        # Fully connected layer
        outputs = self.linear(lstm_out) # outputs shape : (batch_size, caption length, vocab_size)

        return outputs

    ## Greedy search 
    def sample(self, inputs):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        max_sentence = 40
        output = []
        batch_size = inputs.shape[0] # batch_size is 1 at inference, inputs shape : (1, 1, embed_size)
        hidden = self.init_hidden(batch_size) # Get initial hidden state of the LSTM
    
        while max_sentence:
            lstm_out, hidden = self.lstm(inputs, hidden) # lstm_out shape : (1, 1, hidden_size)
            outputs = self.linear(lstm_out)  # outputs shape : (1, 1, vocab_size)
            outputs = outputs.squeeze(1) # outputs shape : (1, vocab_size)
            _, max_indice = torch.max(outputs, dim=1) # predict the most likely next word, max_indice shape : (1)
            
            output.append(max_indice.cpu().numpy()[0].item()) # storing the word predicted
            max_sentence -=1
            if (max_indice == 2):
                # We predicted the <end> word, so there is no further prediction to do
                break
            
            ## Prepare to embed the last predicted word to be the new input of the lstm
            inputs = self.word_embeddings(max_indice) # inputs shape : (1, embed_size)
            inputs = inputs.unsqueeze(1) # inputs shape : (1, 1, embed_size)
            
        return output




    ## Beam search implementation (Attempt)
    def beam_search_sample(self, inputs, beam=3):
        output = []
        batch_size = inputs.shape[0] # batch_size is 1 at inference, inputs shape : (1, 1, embed_size)
        hidden = self.init_hidden(batch_size) # Get initial hidden state of the LSTM
        
        # sequences[0][0] : index of start word
        # sequences[0][1] : probability of the word predicted
        # sequences[0][2] : hidden state related of the last word
        sequences = [[[torch.Tensor([0])], 1.0, hidden]]
        max_len = 20

        ## Step 1
        # Predict the first word <start>
        outputs, hidden = DecoderLstm.get_outputs(self, inputs, hidden)
        _, max_indice = torch.max(outputs, dim=1) # predict the most likely next word, max_indice shape : (1)
        output.append(max_indice.cpu().numpy()[0].item()) # storing the word predicted 
        # inputs = DecoderRNN.get_next_word_input(self, max_indice)
        
        
        l = 0
        while len(sequences[0][0]) < max_len: 
            print("l:", l)
            l+= 1
            temp = []
            for seq in sequences:
#                 print("seq[0]: ", seq[0])
                inputs = seq[0][-1] # last word index in seq
                inputs = inputs.type(torch.cuda.LongTensor)
                print("inputs : ", inputs)
                # Embed the input word
                inputs = self.word_embeddings(inputs) # inputs shape : (1, embed_size)
                inputs = inputs.unsqueeze(1) # inputs shape : (1, 1, embed_size) 
                
                # retrieve the hidden state
                hidden = seq[2]
                
                preds, hidden = DecoderLstm.get_outputs(self, inputs, hidden)

                # Getting the top <beam_index>(n) predictions
                softmax_score = F.log_softmax(outputs, dim=1) # Define a function to sort the cumulative score
                sorted_score, indices = torch.sort(-softmax_score, dim=1)
                word_preds = indices[0][:beam]
                best_scores = sorted_score[0][:beam]

                # Creating a new list so as to put them via the model again
                for i, w in enumerate(word_preds):
#                     print("seq[0]: ", seq[0][0][:].cpu().numpy().item())
                    next_cap, prob = seq[0][0].cpu().numpy().tolist(), seq[1]
                    
                    next_cap.append(w)
                    print("next_cap : ", next_cap)
                    prob *best_scores[i].cpu().item()
                    temp.append([next_cap, prob])

            sequences = temp
            # Order according to proba
            ordered = sorted(sequences, key=lambda tup: tup[1])

            # Getting the top words
            sequences = ordered[:beam]
            print("sequences: ", sequences)

    def get_outputs(self, inputs, hidden):
        lstm_out, hidden = self.lstm(inputs, hidden) # lstm_out shape : (1, 1, hidden_size)
        outputs = self.linear(lstm_out)  # outputs shape : (1, 1, vocab_size)
        outputs = outputs.squeeze(1) # outputs shape : (1, vocab_size)

        return outputs, hidden

    def get_next_word_input(self, max_indice):
        ## Prepare to embed the last predicted word to be the new input of the lstm
        inputs = self.word_embeddings(max_indice) # inputs shape : (1, embed_size)
        inputs = inputs.unsqueeze(1) # inputs shape : (1, 1, embed_size)

        return inputs   


class BeamSearch(nn.Module):
    """Class performs the caption generation using Beam search.
    
    Attributes:
    ----------
    - decoder - trained Decoder of captioning model
    - features - feature map outputed from Encoder
    
    Returns:
    --------
    - sentence - generated caption
    - final_scores - cummulative scores for produced sequences
    """
    def __init__(self, decoder, features, k, max_sentence):
        super().__init__()
        self.k = k
        self.max_sentence = max_sentence
        self.decoder = decoder
        self.features = features
        
        self.h = decoder.init_hidden(features)[0]       
        self.start_idx = torch.ones(1).long()
        self.start_score = torch.FloatTensor([0.0]).repeat(k)
        
        # hidden states on the first step for a single word
        self.hiddens = [[[self.h]]*k]
        self.start_input = torch.FloatTensor[[self.start_idx], self.start_score]
        self.complete_seqs = [list(), list()]
        # track the step
        self.step = 0
        
        
    def beam_search_step(self):
        """Function performs a single step of beam search, returning start input"""
        top_idx_temp = []
        top_score_temp = []
        hiddens_temp = []
        self = self.to(device)
        
        for i, w in enumerate(self.start_input[0][-1].to(device)):
            hidden_states = self.hiddens[self.step][i]
            if self.step==0:   
                h = hidden_states[0].unsqueeze(0)
            else:
                h = hidden_states[0]    

            # scoring stays with the same dimensions
            embedded_word = self.decoder.embed(w.view(-1))
            features = self.decoder.feature_fc(self.features)
            context, atten_weight = self.decoder.attention(features, h)
            # input_concat shape at time step t = (batch, embedding_dim + hidden_dim)
            input_concat = torch.cat([embedded_word, context],  dim = 1)
            
            h = self.decoder.gru(input_concat, h)
            output = self.decoder.fc(h)
            scoring = F.log_softmax(output, dim=1)
            top_scores, top_idx = scoring[0].topk(self.k)
        
            top_cum_score = top_scores + self.start_input[1][i]
            # append top indices and scores
            top_idx_temp.append(top_idx.view(-1, self.k))
            top_score_temp.append(top_cum_score.view(-1, self.k))
            # append hidden states
            hiddens_temp.append([h])
        self.hiddens.append(hiddens_temp)
            
        # concatinate temp lists
        top_idx_temp = torch.cat(top_idx_temp, dim =0)
        top_score_temp = torch.cat(top_score_temp, dim =0)
        cum_score = top_score_temp
        
        top_cum_scores = self.get_cummulative_score(cum_score)
        ready_idx, tensor_positions = self.get_ready_idx(top_cum_scores, 
                                                         top_idx_temp,
                                                         cum_score)
        row_pos = self.get_positions(tensor_positions)
        # update the attributes
        self.update_start_input(ready_idx, row_pos, top_cum_scores)
        self.update_hiddens(row_pos)
        self.update_step()
            
        # step == 1 means we have generated the hiddens from <start> word and outputed k first words
        # we use them to generate k second words
        if self.step == 1:
            self.hiddens[self.step] = self.hiddens[self.step] * self.k
            self.start_input[0][0] = self.start_input[0][0].view(self.k,-1)
        
        return  self.start_input
    
    def get_cummulative_score(self, cum_score):
        """Getting the top scores and indices from cum_score"""
        top_cum_scores, _ = cum_score.flatten().topk(self.k)
        return top_cum_scores
        
    
    def get_ready_idx(self, top_cum_scores, top_idx_temp, cum_score):
        """Obtain a list of ready indices and their positions"""
        # got the list of top positions 
        tensor_positions = [torch.where(cum_score == top_cum_scores[i]) for i in range(self.k)]
        #tensor_positions[1] = [tensor_positions[:][1][0][0],tensor_positions[:][1][1][0]]
        tensor_positions[0] = ((torch.tensor([tensor_positions[0][:][0][0]])),(torch.tensor([tensor_positions[0][:][1][0]])))
        tensor_positions[1] = ((torch.tensor([tensor_positions[1][:][0][0]])),(torch.tensor([tensor_positions[1][:][1][0]])))
        tensor_positions[2] = ((torch.tensor([tensor_positions[2][:][0][0]])),(torch.tensor([tensor_positions[2][:][1][0]])))
        # it's important to sort the tensor_positions by first entries (rows)
        # because rows represent the sequences: 0, 1 or 2 sequences
        tensor_positions = sorted(tensor_positions, key = lambda x: x[0])
        # get read top k indices that will be our input indices for next iteration
        ready_idx = torch.cat([top_idx_temp[tensor_positions[ix]] for ix in range(self.k)]).view(self.k, -1)
        return ready_idx, tensor_positions
        
        
    def get_positions(self, tensor_positions):
        """Retruns the row positions for tensors"""

        row_pos = [x[0] for x in tensor_positions]
        row_pos = torch.cat(row_pos, dim =0)
        return row_pos
    
    def get_nonend_tokens(self):
        """Get tokens that are not <end>"""
        non_end_token = self.start_input[0][-1] !=2
        return non_end_token.flatten()
        

    def update_start_input(self, ready_idx, row_pos, top_cum_scores):      
        """Returns new input sequences"""
        # construct new sequence with respect to the row positions
        start_input_new = [x[row_pos] for x in self.start_input[0]]
        self.start_input[0] = start_input_new 
        start_score_new = self.start_input[1][row_pos]
        self.start_input[1] = start_score_new
        
        # append new indices and update scoring
        self.start_input[0].append(ready_idx)
        self.start_input[1] = top_cum_scores.detach()
        
    def update_hiddens(self, row_pos):
        """Returns new hidden states"""
        self.hiddens = [[x[i] for i in row_pos.tolist()] for x in self.hiddens]
        
    def update_step(self):
        """Updates step"""
        self.step += 1
    
    def generate_caption(self):
        """Iterates over the sequences and generates final caption"""
        while True:
            # make a beam search step 
            self.start_input = self.beam_search_step()
            non_end_token = self.get_nonend_tokens()

            if (len(non_end_token) != sum(non_end_token).item()) and (sum(non_end_token).item() !=0):
                #prepare complete sequences and scores
                non_end_token = non_end_token
                complete_seq = torch.cat(self.start_input[0], dim =1)[non_end_token !=2]
                complete_score = self.start_input[1][non_end_token !=2]
                self.complete_seqs[0].extend(complete_seq)
                self.complete_seqs[1].extend(complete_score)  
            
                start_input_new = torch.cat(self.start_input[0], dim =1)[non_end_token]
                start_input_new = [x.view(len(x), -1) for x in start_input_new.view(len(start_input_new[0]), -1)]
                start_score_new = self.start_input[1][non_end_token]
                
                self.start_input[0] = start_input_new
                self.start_input[1] = start_score_new
                
                non_end_pos = torch.nonzero(non_end_token).flatten()
                self.update_hiddens(non_end_pos)
            elif (sum(non_end_token).item() ==0):
                # prepare complete sequences and scores
                self= self
                complete_seq = torch.cat(self.start_input[0], dim =1)[non_end_token !=2]
                complete_score = self.start_input[1][non_end_token !=2]
                
                self.complete_seqs[0].extend(complete_seq)
                self.complete_seqs[1].extend(complete_score) 
            else:
                pass
            if (len(self.complete_seqs[0])>=self.k or self.step == self.max_sentence):
                break
        
        return self.get_top_sequence()

                
            
    def get_top_sequence(self):
        """Gets the sentence and final set of scores"""
        lengths = [len(i) for i in self.complete_seqs[0]]
        final_scores = [self.complete_seqs[1][i] / lengths[i] for i in range(len(lengths))]
        if final_scores:
            best_score = np.argmax([i.item() for i in final_scores])
            sentence = self.complete_seqs[0][best_score].tolist()
        else:
            sentence, final_scores   = "", 0.0    
        return sentence, final_scores        