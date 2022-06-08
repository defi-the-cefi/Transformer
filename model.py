import torch
import torch.nn as nn
import numpy as np


# Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, batch_size, sequence_length, input_dim, hidden_dim, layer_dim, nheads, pred_len, output_dim, dropout_prob):
        super().__init__()
        #super(TransformerModel, self).__init__()
        self.input_dim = input_dim
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim
        self.pred_len = pred_len
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.dropout_prob = dropout_prob

        # transforder layers
        # self.encoder = nn.Embedding(self.sequence_length, self.hidden_dim)
        self.positional_encoder = DataEmbedding(c_in=self.input_dim, dim_model=self.hidden_dim, dropout=self.dropout_prob)# , max_len=self.sequence_length)
        # self.pos_encoder = PositionalEncoding(self.hidden_dim, self.dropout_prob)
        self.fc_pre = nn.Linear(self.input_dim * self.sequence_length, self.sequence_length * self.hidden_dim)
        self.trans = torch.nn.Transformer(d_model=self.hidden_dim, nhead=nheads, num_encoder_layers=self.layer_dim, num_decoder_layers=self.layer_dim, dim_feedforward=4*self.hidden_dim, dropout=dropout_prob) #, batch_first=True)
        self.fc_post = nn.Linear(self.hidden_dim, self.pred_len * self.output_dim)
        # self.decoder = nn.Linear(self.hidden_dim, self.sequence_length)

    def generate_square_subsequent_mask(sz: int):# -> nn.Tensor:
        """Generates an upper-triangular matrix of -inf, with zeros on diag."""
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

    def forward(self, x,y):
        # x.shape = batch_size x input_dim x seq_leng
        # nn.transformer takes input in shape batch x seq_leng x encoder_size (N,S,E)
        # src.shape and tgt.shape should be  = [self.batch_size, self.sequence_length,self.hidden_dim]
        # embedding layer before feeding input to transformer layer
        src = self.positional_encoder(x, c_in=self.input_dim, dim_model=self.hidden_dim)
        print('shape of input tensor being sent into transformer', src.shape)
        tgt = self.positional_encoder(y, c_in=self.output_dim, dim_model=self.hidden_dim)
        print('shape of output tensor being sent into transformer', tgt.shape)
        # TokenEmbedding embeds into dim_hidden for us already so we don't need to run it thru fc_pre
        # src = self.fc_pre(x)
        # src = src.view()
#         tgt_mask = self.generate_square_subsequent_mask(tgt.shape[2])
#         print(tgt_mask)
        # tranformer model
        transformer_out = self.trans(src, tgt, tgt_mask)
        print('trans output shape', transformer_out.shape)
        print('should be shape batch x target_len x encoder_size (N,T,E)')
        # outputs tensor of same shape batch x target_len x encoder_size (N,T,E)
#        print('pre-fc out.shape: ', out.shape)
#        out = self.fc(out)
        return transformer_out
