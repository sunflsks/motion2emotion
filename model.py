import torch
from torch import nn

class Transformer(nn.Module):
    def __init__(self, input_dim=36, d_model=256, seq_len=64, num_heads=4, num_layers=8, output_dim=26):
        super().__init__()
        self.tokenizer = nn.Linear(input_dim, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(1, seq_len, d_model))
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.output_layer = nn.Linear(d_model, output_dim)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x, mask=None):
        x = self.tokenizer(x) + self.positional_encoding
        x = self.dropout(x)
        x = self.encoder(x, src_key_padding_mask=mask)
        x = x.mean(dim=1)
        x = self.norm(x)
        return self.output_layer(x)