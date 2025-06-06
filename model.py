import torch
from torch import nn
import torch.nn.functional as F

class Transformer(nn.Module):
    def __init__(self, input_dim=12, d_model=192, seq_len=64, num_heads=3, num_layers=6, dim_feedforward=256, output_dim=26):
        super().__init__()

        # projects raw coordinates into the model's vector space (linear projection)
        self.tokenizer = nn.Linear(input_dim, d_model)

        # normalize projection
        self.input_norm = nn.LayerNorm(d_model)

        # dropout
        self.input_dropout = nn.Dropout(0.1)

        # classification token
        self.class_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.class_norm = nn.LayerNorm(d_model)

        # positional embeddings
        self.positional_encoding = nn.Parameter(torch.randn(1, seq_len + 1, d_model))

        # the magic
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            batch_first=True,
            dim_feedforward=dim_feedforward
        )

        # transformer
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        # classify token to logits
        self.output_layer = nn.Linear(d_model, output_dim)

    def forward(self, x, mask=None):
        # x.shape = (batch_size, seq_len, 36 // 18 x 2, 18 joints and X, Y dims // )
        batch_size = x.shape[0] # prob ~64

        if mask is not None:
            dev = mask.device
            false_tensor = torch.tensor([False]).to(dev).expand(batch_size, -1)
            mask = torch.cat([false_tensor, mask], dim=1)


        x = self.tokenizer(x) # project into d-dimensional space
        x = self.input_norm(x) # normalize
        x = self.input_dropout(x)

        class_token = self.class_token.expand(batch_size, -1, -1) # get class token
        x = torch.cat([class_token, x], dim=1) # add class token to sequence

        x = x + self.positional_encoding[:, : x.shape[1], :] # add positional encodings

        x = self.transformer(x, src_key_padding_mask=mask) # transform

        x = x[:, 0, :] # isolate class token
        x = self.class_norm(x)

        return self.output_layer(x)