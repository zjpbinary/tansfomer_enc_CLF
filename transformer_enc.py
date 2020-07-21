import math
import torch
import torch.nn as nn

class LearnedPositionEncoding(nn.Embedding):
    def __init__(self, d_model, dropout = 0.5, max_len = 5000):
        super().__init__(max_len, d_model)
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, x):
        #因为分批处理，添加一维度
        weight = self.weight.data.unsqueeze(1)
        x = x + weight[:x.size(0), :]
        return self.dropout(x)
class S2sTransformer(nn.Module):
    def __init__(self, vocab_size, label_size, max_len, pos_enc, d_model, nhead = 8, num_encoder_layers = 1,
                 dim_feedforward = 128, dropout = 0.5):
        super(S2sTransformer, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc = pos_enc(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        self.decoder = nn.Linear(d_model*max_len, label_size)
    def forward(self, src, src_mask = None):
        src = self.embedding(src)
        src = self.pos_enc(src)
        memory = self.encoder(src, src_mask).transpose(0, 1)
        memory = torch.reshape(memory, (memory.size(0), -1))
        output = self.decoder(memory)
        return output



