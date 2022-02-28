import torch
from torch import nn
from  transformers.models.bert import modeling_bert

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VLBertEmbeddings(modeling_bert.BertEmbeddings):
    def __init__(self, config):
        super(VLBertEmbeddings, self).__init__(config)

        self.image_embed = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob),
            )
            



    def forward(self, images_features, input_token_ids, token_type_ids, position_ids):
        image_embed = self.image_embed(images_features)
        words_embeddings = self.word_embeddings(input_token_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        words_embeddings = torch.cat((image_embed, words_embeddings), dim=1)
        position_embeddings = torch.cat((image_embed, position_embeddings), dim=1)
        token_type_embeddings = torch.cat((image_embed, token_type_embeddings), dim=1)
        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        return self.dropout(self.LayerNorm(embeddings))



class Generator(modeling_bert.BertPreTrainedModel):
    def __init__(self, config):
        super(Generator, self).__init__(config)

        self.encoder = modeling_bert.BertEncoder(config)
        self.classifier = modeling_bert.BertLMPredictionHead(config)
        self.embedding_layer = VLBertEmbeddings(config)
        self.head_mask = [None] * config.num_hidden_layers

        self.apply(self._init_weights)

    def load_weights(self, path):
        state_dict = torch.load(path, map_location=torch.device("cpu"))
        if 'model' in state_dict.keys():
            state_dict = state_dict['model']
        self.load_state_dict(state_dict, strict=False)
        del state_dict

    def forward(self, images_features, masked_token_ids, token_type_ids, position_ids, attention_mask, padding):
        embeddings = self.embedding_layer(images_features, masked_token_ids, token_type_ids, position_ids)
  
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        attention_mask = torch.cat((attention_mask, padding), dim=3)
        attention_mask = (1.0 - attention_mask) * -10000.0
        hidden_states = self.encoder(embeddings, attention_mask, self.head_mask)[0]
        return self.classifier(hidden_states)


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, weight=None, smoothing=0.0):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.classes = classes
        self.weight = weight

    def forward(self, pred, target):
        """
        Args:
            pred: (N, C), float
            target: (N,), long, values in [0, C-1]
        """
        if self.weight is None:
            self.weight = torch.ones(self.classes, dtype=torch.float32,
                                     device=target.device)
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.classes - 1))
            try:
                true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
            except IndexError:
                true_dist = true_dist
                

        weight = self.weight[target].to(device)
        weighted_loss = torch.sum(-true_dist * pred, dim=-1) * weight

        return torch.mean(weighted_loss) * weight.numel() / weight.sum()