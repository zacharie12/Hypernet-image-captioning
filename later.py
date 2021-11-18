class DecoderWithAttention(nn.Module):
    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=2048, dropout=0.5,
                 embedding_matrix=None, train_embd=True):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(DecoderWithAttention, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  # attention network

        self.embedding = embedding_layer(num_embeddings=vocab_size, embedding_dim=embed_dim,
                                         embedding_matrix=embedding_matrix, trainable=train_embd)
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state [b, decoder_dim]
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        Forward propagation.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # Flatten image
        # [b, num_pixels, encoder_dim]
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)
        num_pixels = encoder_out.size(1)

        # Sort input data by decreasing lengths; why? apparent below
        # [b, 1] -> [b], [b]
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind].long() 

        # Embedding
        # [b, max_len, embed_dim]
        embeddings = self.embedding(encoded_captions)

        # Initialize LSTM state
        # [b, decoder_dim]
        h, c = self.init_hidden_state(encoder_out)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word predicion scores and alphas
        # [b, max_len, vocab_size]
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(encoder_out.device)
        # [b, num_pixels, vocab_size]
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(encoder_out.device)

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            # [b, encoder_dim], [b, num_pixels] -> [batch_size_t, encoder_dim], [batch_size_t, num_pixels]
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],
                                                                h[:batch_size_t])
            # [batch_size_t, encoder_dim]
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar,
            attention_weighted_encoding = gate * attention_weighted_encoding
            # [batch_size_t, decoder_dim]
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))
            # [batch_size_t, vocab_size]
            preds = self.fc(self.dropout(h))
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind

    def sample(self, encoder_out, startseq_idx, endseq_idx=-1, max_len=40, return_alpha=False):
        """
        Samples captions in batch for given image features (Greedy search).
        :param encoder_out = [b, enc_image_size, enc_image_size, 2048]
        :return [b, max_len]
        """
        enc_image_size = encoder_out.size(1)
        encoder_dim = encoder_out.size(3)
        batch_size = encoder_out.size(0)

        # decoder = self
        # [b, enc_image_size, enc_image_size, 2048] -> [b, num_pixels, 2048]
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)
        # [b, num_pixels, ]
        h, c = self.init_hidden_state(encoder_out)

        sampled_ids = []  # list of [b,]
        alphas = []

        # [b, 1]
        prev_timestamp_words = torch.LongTensor([[startseq_idx]] * batch_size).to(encoder_out.device)
        for i in range(max_len):
            # [b, 1] -> [b, embed_dim]
            embeddings = self.embedding(prev_timestamp_words).squeeze(1)
            # ([b, encoder_dim], [b, num_pixels])
            awe, alpha = self.attention(encoder_out, h)
            # [b, enc_image_size, enc_image_size] -> [b, 1, enc_image_size, enc_image_size]
            alpha = alpha.view(-1, enc_image_size, enc_image_size).unsqueeze(1)

            # [b, embed_dim]
            gate = self.sigmoid(self.f_beta(h))  # gating scalar
            # [b, embed_dim]
            awe = gate * awe

            # ([b, decoder_dim], )
            h, c = self.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))
            # [b, vocab_size]
            predicted_prob = self.fc(h)
            # [b]
            predicted = predicted_prob.argmax(1)

            sampled_ids.append(predicted)
            alphas.append(alpha)

            # [b] -> [b, 1]
            prev_timestamp_words = predicted.unsqueeze(1)
        # [b, max_len]
        sampled_ids = torch.stack(sampled_ids, 1)
        return (sampled_ids, torch.cat(alphas, 1)) if return_alpha else sampled_ids       


    def sample_val(self, encoder_out, startseq_idx, caption_lengths, encoded_captions, endseq_idx=-1, return_alpha=False):
 
        enc_image_size = encoder_out.size(1)
        encoder_dim = encoder_out.size(3)
        batch_size = encoder_out.size(0)
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        decode_lengths = (caption_lengths - 1).tolist()
        encoded_captions = encoded_captions[sort_ind].long() 

        # decoder = self
        # [b, enc_image_size, enc_image_size, 2048] -> [b, num_pixels, 2048]
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)
        # [b, num_pixels, ]
        h, c = self.init_hidden_state(encoder_out)

        sampled_ids = []  # list of [b,]
        alphas = []

        vocab_size = self.vocab_size
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(encoder_out.device)

        # [b, 1]
        prev_timestamp_words = torch.LongTensor([[startseq_idx]] * batch_size).to(encoder_out.device)
        for t in range(max(decode_lengths)):
            batch_size_t = batch_size
            embeddings = self.embedding(prev_timestamp_words).squeeze(1)
            # ([b, encoder_dim], [b, num_pixels])
            awe, alpha = self.attention(encoder_out, h)
            # [b, enc_image_size, enc_image_size] -> [b, 1, enc_image_size, enc_image_size]
            alpha = alpha.view(-1, enc_image_size, enc_image_size).unsqueeze(1)

            # [b, embed_dim]
            gate = self.sigmoid(self.f_beta(h))  # gating scalar
            # [b, embed_dim]
            awe = gate * awe

            # ([b, decoder_dim], )
            h, c = self.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))
            # [b, vocab_size]
            predicted_prob = self.fc(h)
            predictions[:batch_size_t, t, :] = predicted_prob
            # [b]
            predicted = predicted_prob.argmax(1)

            sampled_ids.append(predicted)
            alphas.append(alpha)

            # [b] -> [b, 1]
            prev_timestamp_words = predicted.unsqueeze(1)
        # [b, max_len]
        sampled_ids = torch.stack(sampled_ids, 1)
        if True:
            return predictions, decode_lengths, encoded_captions, (sampled_ids, torch.cat(alphas, 1)) if return_alpha else sampled_ids      
        else:
            return (sampled_ids, torch.cat(alphas, 1)) if return_alpha else sampled_ids   




class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, dropout=False):
        super(DecoderRNN, self).__init__()
        
        # define the properties
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.dropout=dropout
        self.num_layers = num_layers
        with open("data/vocab.pkl", 'rb') as f:
            self.vocab =  pickle.load(f)
        
        # lstm cell
        self.lstm_cell = nn.LSTMCell(input_size=embed_size, hidden_size=hidden_size)
        self.layers = None
        if num_layers > 1:
            self.layers = nn.ModuleList([nn.LSTMCell(input_size=hidden_size, hidden_size=hidden_size) for layer in range(num_layers-1)])
        # output fully connected layer
        self.fc_out = nn.Linear(in_features=self.hidden_size, out_features=self.vocab_size)
    
        # embedding layer
        self.embed = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embed_size)
    
        # activations
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, features, captions, teacher_forcing=True):
        # batch size
        batch_size = features.size(0)
        
        # init the hidden and cell states to zeros
        hidden_state = torch.zeros((batch_size, self.hidden_size))
        hidden_state = hidden_state.type_as(features)
        cell_state = torch.zeros((batch_size, self.hidden_size))
        cell_state = cell_state.type_as(features)
    
        # define the output tensor placeholder
        outputs = torch.empty((batch_size, captions.size(1), self.vocab_size))
        outputs = outputs.type_as(features)

        # embed the captions
        captions_embed = self.embed(captions)
        if self.dropout:
            captions_embed = F.dropout(captions_embed, 0.5)
        out = None

        # pass the caption word by word
        for t in range(captions.size(1)):

            # for the first time step the input is the feature vector
            if t == 0:
                hidden_state, cell_state = self.lstm_cell(features, (hidden_state, cell_state))
                if self.layers:
                    for layer in self.layers:
                        hidden_state, cell_state = layer(hidden_state, (hidden_state, cell_state))
            # for the 2nd+ time step, using teacher forcer
            else:
                if teacher_forcing:
                    hidden_state, cell_state = self.lstm_cell(captions_embed[:, t-1, :], (hidden_state, cell_state))
                    if self.layers:
                        for layer in self.layers:
                            hidden_state, cell_state = layer(hidden_state, (hidden_state, cell_state))
                else:
                    pred = self.softmax(out)
                    #chosen_words = torch.argmax(pred, dim=1)
                    chosen_words = torch.multinomial(pred, 1).t()[0]
                    embed_words = self.embed(chosen_words)
                    if self.dropout:
                        embed_words = F.dropout(embed_words, 0.5)
                    hidden_state, cell_state = self.lstm_cell(embed_words, (hidden_state, cell_state))
                    if self.layers:
                        for layer in self.layers:
                            hidden_state, cell_state = layer(hidden_state, (hidden_state, cell_state))
                    '''
                    gt_word = captions[:, t-1]
                    pred_word = chosen_words
                    st_gt = cap_to_text_gt(gt_word, self.vocab)
                    st_pred = cap_to_text_gt(pred_word, self.vocab)
                    '''
                    
            # output of the attention mechanism
            out = self.fc_out(hidden_state)
            prediction = self.softmax(out)
            prediction = out
            chosen_word = torch.argmax(prediction, dim=1)
            gt_word_out = captions[:, t]
            a = prediction.view(-1)
            b= gt_word_out.view(-1)
            loss = F.cross_entropy(prediction.view(-1, 9684), gt_word_out.view(-1))

            out_word_gt = cap_to_text_gt(gt_word_out, self.vocab)
            out_word_pred = cap_to_text_gt(chosen_word, self.vocab)

            # build the output tensor
            outputs[:, t, :] = out
  
        return outputs

    def infer(self, features, max_len=50):
        # batch size
        batch_size = features.size(0)
        
        # init the hidden and cell states to zeros
        hidden_state = torch.zeros((batch_size, self.hidden_size))
        hidden_state = hidden_state.type_as(features)
        cell_state = torch.zeros((batch_size, self.hidden_size))
        cell_state = cell_state.type_as(features)
    
        # define the output tensor placeholder
        outputs = torch.empty((batch_size, max_len, self.vocab_size))
        outputs = outputs.type_as(features)
        out = None
        # pass the caption word by word
        for t in range(max_len):

            # for the first time step the input is the feature vector
            if t == 0:
                hidden_state, cell_state = self.lstm_cell(features, (hidden_state, cell_state))
                
            # for the 2nd+ time step, using teacher forcer
            else:
                chosen_words = torch.argmax(out, dim=1)
                embed_words = self.embed(chosen_words)
                hidden_state, cell_state = self.lstm_cell(embed_words, (hidden_state, cell_state))
            
            # output of the attention mechanism
            out = self.softmax(self.fc_out(hidden_state))
            
            # build the output tensor
            outputs[:, t, :] = out
    
        return outputs


class DecoderGRU(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, dropout=False):
        super(DecoderGRU, self).__init__()
        
        # define the properties
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.dropout=dropout
        self.num_layers = num_layers
        with open("data/vocab.pkl", 'rb') as f:
            self.vocab =  pickle.load(f)
        
        # gru cell
        self.lstm_cell = nn.GRUCell(input_size=embed_size, hidden_size=hidden_size)
        self.layers = None
        if num_layers > 1:
            self.layers = nn.ModuleList([nn.GRUCell(input_size=hidden_size, hidden_size=hidden_size) for layer in range(num_layers-1)])
        # output fully connected layer
        self.fc_out = nn.Linear(in_features=self.hidden_size, out_features=self.vocab_size)
    
        # embedding layer
        self.embed = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embed_size)
    
        # activations
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, features, captions, teacher_forcing=True):
        # batch size
        batch_size = features.size(0)
        # init the hidden and cell states to zeros
        hidden_state = torch.rand(size=(batch_size, self.hidden_size))
        hidden_state = hidden_state.type_as(features)

    
        # define the output tensor placeholder
        outputs = torch.empty((batch_size, captions.size(1), self.vocab_size))
        outputs = outputs.type_as(features)

        # embed the captions
        captions_embed = self.embed(captions)

        out = None

        # pass the caption word by word
        for t in range(captions.size(1)):

            # for the first time step the input is the feature vector
            if t == 0:
                hidden_state = self.lstm_cell(features, hidden_state)
                if self.layers:
                    for layer in self.layers:
                        hidden_state = layer(hidden_state, hidden_state)
            # for the 2nd+ time step, using teacher forcer
            else:
                if teacher_forcing:
                    hidden_state = self.lstm_cell(captions_embed[:, t-1, :], hidden_state)
                    if self.layers:
                        for layer in self.layers:
                            hidden_state = layer(hidden_state, hidden_state)
                else:
                    pred = self.softmax(out)
                    #chosen_words = torch.argmax(pred, dim=1)
                    chosen_words = torch.multinomial(pred, 1).t()[0]
                    #chosen_words = sample_multinomial_topk(pred, len(self.vocab)).to(device)
                    embed_words = self.embed(chosen_words)
                    if self.dropout:
                        embed_words = F.dropout(embed_words, 0.5)
                    hidden_state  = self.lstm_cell(embed_words, (hidden_state))
                    if self.layers:
                        for layer in self.layers:
                            hidden_state = layer(hidden_state, (hidden_state))
                    '''
                    gt_word = captions[:, t-1]
                    pred_word = chosen_words
                    st_gt = cap_to_text_gt(gt_word, self.vocab)
                    st_pred = cap_to_text_gt(pred_word, self.vocab)
                    '''
                    
            # output of the attention mechanism
            out = self.fc_out(hidden_state)
            prediction = self.softmax(out)
            prediction = out
            chosen_word = torch.argmax(prediction, dim=1)
            gt_word_out = captions[:, t]
            a = prediction.view(-1)
            b= gt_word_out.view(-1)
            loss = F.cross_entropy(prediction.view(-1, 9684), gt_word_out.view(-1))

            out_word_gt = cap_to_text_gt(gt_word_out, self.vocab)
            out_word_pred = cap_to_text_gt(chosen_word, self.vocab)

            # build the output tensor
            outputs[:, t, :] = out
  
        return outputs

    def infer(self, features, max_len=50):
        # batch size
        batch_size = features.size(0)
        
        # init the hidden and cell states to zeros
        hidden_state = torch.rand(size=(batch_size, self.hidden_size))
        hidden_state = hidden_state.type_as(features)
    
        # define the output tensor placeholder
        outputs = torch.empty((batch_size, max_len, self.vocab_size))
        outputs = outputs.type_as(features)
        out = None
        # pass the caption word by word
        for t in range(max_len):

            # for the first time step the input is the feature vector
            if t == 0:
                hidden_state = self.lstm_cell(features, hidden_state)
                
            # for the 2nd+ time step, using teacher forcer
            else:
                chosen_words = torch.argmax(out, dim=1)
                embed_words = self.embed(chosen_words)
                hidden_state = self.lstm_cell(embed_words, hidden_state)
            
            # output of the attention mechanism
            out = self.softmax(self.fc_out(hidden_state))
            
            # build the output tensor
            outputs[:, t, :] = out
    
        return outputs

class Lstm_net(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(Lstm_net, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embed_size= embed_size
        self.drop_prob= 0.2
        self.vocabulary_size = vocab_size
        with open("data/vocab.pkl", 'rb') as f:
            self.vocab =  pickle.load(f)
        self.lstm = nn.LSTM(self.embed_size, self.hidden_size , self.num_layers,batch_first=True)
        self.dropout = nn.Dropout(self.drop_prob)
        self.embed = nn.Embedding(self.vocabulary_size, self.embed_size)
        self.linear = nn.Linear(hidden_size, self.vocabulary_size)
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)
    
    def forward(self, features, captions):
        embeddings = self.embed(captions)
        features = features.unsqueeze(1)
        embeddings = torch.cat((features, embeddings[:, :-1,:]), dim=1)
        hiddens, c = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs
