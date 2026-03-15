import torch
from torch import nn

class CrossTransformer(nn.Module):
    def __init__(self):
        super(CrossTransformer, self).__init__()
        self.cross_attention = nn.MultiheadAttention(256, 4, batch_first=True)
        self.ffn = nn.Sequential(nn.LayerNorm(256),
                                 nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 256), 
                                 nn.LayerNorm(256),
                                 )

    def forward(self, query, key, mask=None):
        value = key
        mask[:, 0] = False
        attention_output, _ = self.cross_attention(query, key, value, key_padding_mask=mask)
        output = self.ffn(attention_output)

        return output

class SelfTransformer(nn.Module):
    def __init__(self):
        super(SelfTransformer, self).__init__()
        self.self_attention = nn.MultiheadAttention(256, 4, batch_first=True)
        self.ffn = nn.Sequential(nn.LayerNorm(256),
                                 nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 256), 
                                 nn.LayerNorm(256),
                                 )

    def forward(self, input, mask=None):
        attention_output, _ = self.self_attention(input, input, input, key_padding_mask=mask)
        output = self.ffn(attention_output)

        return output

class SubGraph(nn.Module):
    def __init__(self, c_in, hidden_size=256):
        super(SubGraph, self).__init__()
        self.hidden_size = hidden_size
        self.c_in = c_in

        self.layer = nn.Sequential(nn.Linear(c_in, hidden_size), 
                                 nn.ReLU(), nn.Linear(hidden_size, hidden_size), 
                                 )
        self.attention = SelfTransformer()


    def forward(self, x):
        # input shape (B, N_h, c_in) or (B, N_n, N_h, c_in)

        h = x.reshape(-1, x.size(-2), x.size(-1)) # (_, N_h, c_in)

        mask = torch.eq(torch.sum(h, -1), 0) # (_, N_h)
        mask[:, -1] = False 

        h = self.layer(h)
        h = self.attention(h, mask=mask)

        h = h.reshape(x.size(0), -1, x.size(-2), self.hidden_size)
        # print(h.shape)
        output, _ = h.max(2)#torch.max(h, 2) #[0]
        #print(output.shape)
        return output
