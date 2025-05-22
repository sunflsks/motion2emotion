import torch
from torch import nn

class Transformer(nn.Module):
    def __init__(self, input_dim=36, d_model=192, seq_len=64, num_heads=3, num_layers=6, dim_feedforward=256, output_dim=26):
        super().__init__()

        # projects raw coordinates into the model's vector space (linear projection)
        self.tokenizer = nn.Linear(input_dim, d_model)

        # classification token
        self.class_token = nn.Parameter(torch.randn(1, 1, d_model))

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
        dev = mask.device
        batch_size = x.shape[0] # prob ~64

        false_tensor = torch.tensor([False]).to(dev).expand(batch_size, -1)
        mask = torch.cat([false_tensor, mask], dim=1)

        x = self.tokenizer(x) # project into d-dimensional space
        class_token = self.class_token.expand(batch_size, -1, -1) # get class token
        x = torch.cat([class_token, x], dim=1) # add class token to sequence
        x = x + self.positional_encoding[:, : x.shape[1], :] # add positional encodings
        x = self.transformer(x, src_key_padding_mask=mask) # transform
        x = x[:, 0, :] # isolate class token
        return self.output_layer(x)
