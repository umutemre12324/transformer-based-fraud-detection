import torch
import torch.nn as nn


class Transformer(nn.Module):

    def __init__(
        self,
        input_dim=5,
        model_dim=64,
        num_head=4,
        num_layer=2,
        num_classes=3,
        dropout=0.1
    ):
        super(Transformer, self).__init__()

        self.input_projection = nn.Linear(input_dim, model_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_head,
            dim_feedforward=128,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layer
        )

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Linear(model_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        # x: (batch_size, 5, 20)
        x = x.transpose(1, 2)

        # (batch_size, 20, 5)
        x = self.input_projection(x)

        # (batch_size, 20, 64)
        x = self.transformer_encoder(x)

        # Zaman adımları boyunca ortalama
        x = x.mean(dim=1)

        # Sınıflandırma
        x = self.dropout(x)
        x = self.classifier(x)

        return x


