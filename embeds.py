import torch
import torch.nn as nn
import math



class PositionalEmbedding(nn.Module):
    def __init__(self, dim_model, max_len=5000):
        # max_len determines how far the position can have an effect on a token (window)
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, dim_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, dim_model, 2).float() * -(math.log(10000.0) / dim_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = self.pe[:, :x.size(1)]
        print('pe shape', x.shape)
        return x

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, dim_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=dim_model,
                                   kernel_size=3, padding=padding, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1,2)
        print('conv embed shape', x.shape)
        return x


class DataEmbedding(nn.Module):
    def __init__(self, c_in, dim_model, dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, dim_model=dim_model)
        self.position_embedding = PositionalEmbedding(dim_model=dim_model) #, max_len)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        print('final embedding shape', x.shape)
        return self.dropout(x)
