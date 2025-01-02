import torch
import math
from torch import nn
from torch.nn import functional as F
import torchinfo


class CausalConv1d(torch.nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias)

        self.__padding = (kernel_size - 1) * dilation

    def forward(self, x):
        return super(CausalConv1d, self).forward(F.pad(x, (self.__padding, 0)))


class ContextEmbedding(torch.nn.Module):
    def __init__(self, in_channels=1, embedding_size=256, kernel_size=5):
        super().__init__()
        self.causal_convolution = CausalConv1d(in_channels, embedding_size, kernel_size=kernel_size)

    def forward(self, x):
        x = self.causal_convolution(x)
        return F.tanh(x)

class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1)  # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model)
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)
        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, x):
        # print(x.shape,self.pos_encoding[:x.size(1),:].shape)
        pos_embeddings = self.pos_encoding[:x.size(1),:].unsqueeze(0)
        # print(pos_embeddings.shape)
        x = x + pos_embeddings
        return self.dropout(x)



class Transformer(nn.Module):
    def __init__(self,num_tokens, dim_model,num_heads, dropout_p):
        super().__init__()
        self.embedding = nn.Embedding(num_tokens, dim_model)
        self.positional_encoder = PositionalEncoding(dim_model=dim_model, dropout_p=dropout_p, max_len=128)
        self.transformer = nn.Transformer(dim_model=dim_model, nhead=num_heads,
                                          num_encoder_layers=3, num_decoder_layers=3,
                                          dim_feedforward=4*dim_model, dropout_p=dropout_p)
        self.out = nn.Linear(dim_model, num_tokens)

    def forward(self, src, trg):
        src = self.positional_encoder(self.embedding(src) * math.sqrt(self.dim_model))
        trg = self.positional_encoder(self.embedding(src) * math.sqrt(self.dim_model))
        src = src.permute(1, 0, 2)
        tgt = trg.permute(1, 0, 2)
        transformer_out = self.transformer(src, tgt)
        out = self.out(transformer_out)
        return out


class RNABERT(nn.Module):
    def __init__(self, dim_model, num_heads, dropout_p):
        super().__init__()
        self.dim_model = dim_model
        self.embedding = nn.Embedding(6, dim_model)
        self.positional_encoder = PositionalEncoding(dim_model=dim_model, dropout_p=dropout_p, max_len=128)
        self.encoder_layer = nn.TransformerEncoderLayer(dim_model,nhead=num_heads
                                                        ,dim_feedforward=4*dim_model, dropout=dropout_p)
        self.norm = nn.LayerNorm(dim_model)
        self.encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer,num_layers=3, norm=self.norm)
        self.out = nn.Linear(dim_model, 1)

    def forward(self, x):   # N
        x = self.positional_encoder(self.embedding(x) * math.sqrt(self.dim_model))
        x = x.permute(1, 0, 2)
        x = self.encoder_layer(x)
        output = self.out(x)
        return output

class ConvBERT(nn.Module):
    def __init__(self,in_channels, dim_model, kernel_size, num_heads, dropout_p,max_len=10):
        super().__init__()
        self.dim_model = dim_model
        self.embedding = ContextEmbedding(in_channels,dim_model,kernel_size=kernel_size)
        self.positional_encoder = PositionalEncoding(dim_model=dim_model, dropout_p=dropout_p, max_len=max_len)
        self.encoder_layer = nn.TransformerEncoderLayer(dim_model,nhead=num_heads
                                                        ,dim_feedforward=4*dim_model, dropout=dropout_p)
        self.norm = nn.LayerNorm(dim_model)
        self.encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer,num_layers=3, norm=self.norm)
        self.out = nn.Linear(dim_model*max_len, 1)

    def forward(self, x):   # N
        bs = x.shape[0]
        x= self.embedding(x.permute(0,2,1)) # N,C,D
        x = self.positional_encoder(x.permute(0,2,1) * math.sqrt(self.dim_model))
        x = x.permute(1, 0, 2)
        x = self.encoder_layer(x)
        output = self.out(x.reshape(bs, -1))
        return output


if __name__ == '__main__':
    # x = torch.randint(0,6,(4, 118))
    # model = RNABERT(dim_model=128, num_heads=8, dropout_p=0.2)
    x = torch.rand((4,118,4))
    model = ConvBERT(in_channels=4, dim_model=128,kernel_size=5, num_heads=8,
                     dropout_p=0.2,max_len=118)
    torchinfo.summary(model,input_data=x)





