import torch
from torch import nn
from torch.nn import functional as F

class BahdanauAttention(nn.Module):
    """ Class performs Additive Bahdanau Attention.
    Source: https://arxiv.org/pdf/1409.0473.pdf
    """
    def __init__(self, num_features, hidden_dim, output_dim = 1):
        super(BahdanauAttention, self).__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        # fully-connected layer to learn first weight matrix Wa
        self.W_a = nn.Linear(self.num_features, self.hidden_dim)
        # fully-connected layer to learn the second weight matrix Ua
        self.U_a = nn.Linear(self.hidden_dim, self.hidden_dim)
        # fully-connected layer to produce score (output), learning weight matrix va
        self.v_a = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, features, decoder_hidden):
        """
        Arguments:
        ----------
        - features - features returned from Encoder
        - decoder_hidden - hidden state output from Decoder
        Returns:
        ---------
        - context - context vector with a size of (1,2048)
        - atten_weight - probabilities, express the feature relevance
        """
        # add additional dimension to a hidden (need for summation later)
        decoder_hidden = decoder_hidden.unsqueeze(1)
        atten_1 = self.W_a(features)
        atten_2 = self.U_a(decoder_hidden)
        # apply tangent to combined result from 2 fc layers
        atten_tan = torch.tanh(atten_1+atten_2)
        # one score corresponds to one Encoder's output
        atten_score = self.v_a(atten_tan)
        atten_weight = F.softmax(atten_score, dim = 1)
        # first, we will multiply each vector by its softmax score
        # next, we will sum up this vectors, producing the attention context vector
        context = torch.sum(atten_weight * features,
                           dim = 1)
        atten_weight = atten_weight.squeeze(dim=2)
        return context, atten_weight