import torch
import torch.nn as nn
from Transformer import Transformer

class BigBAT(nn.Module):
    
    def __init__(
        self,
        max_len,
        d_model, # must equal to patch_embedding output dim
        num_classes,
        nhead=2,
        dim_feedforward=32,
        num_layers=2,
        dropout=0.1,
        classifier_dropout=0.3
    ):

        super().__init__()

        assert d_model % nhead == 0, "nheads must divide evenly into d_model"
        
        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(5, 7), stride=(2, 3), padding=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(16, 32, kernel_size=(3, 5), stride=(2, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=5, stride=2, padding=1),

            nn.Conv2d(32, 64, kernel_size=(3, 5), stride=(1, 2), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
    
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_encoder = nn.Parameter(torch.randn(1, max_len + 1, d_model))

        self.dropout = nn.Dropout(classifier_dropout)
        
        self.transformer_encoder = Transformer(
            dim=d_model,
            depth=num_layers,
            heads=nhead,
            dim_head=16,
            mlp_dim=dim_feedforward,
            dropout=dropout)
        
        self.classifier = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, num_classes))
        
        self.d_model = d_model

    def forward(self, x):
        # preprocessing
        x = torch.abs(torch.stft(x, n_fft=512, window=torch.hann_window(window_length=512).cuda(), return_complex=True)) # FFT
        x = 20 * torch.log10(x / torch.max(x)) # amplitude to db
        x = torch.abs(x - x.mean(dim=2, keepdim=True).repeat((1, 1, x.shape[2]))) # noise filter
        x = x.transpose(dim0=2, dim1=1)
        x = x.unfold(dimension=1, size=44, step=22).permute((0, 1, 3, 2)) # patches
        x[torch.isinf(x)] = 0
        x[torch.isnan(x)] = 0
        
        b, n, w, h = x.shape
        x = x.reshape((b * n, 1, w, h))
        x = self.to_patch_embedding(x)
        x = x.reshape((b, n, self.d_model))
        
        #cls = einops.repeat(self.cls_token, '1 n d -> b n d', b=b)
        cls = self.cls_token.repeat((b, 1, 1))
        x = torch.cat((cls, x), dim=1)
        x += self.pos_encoder
        
        x = self.dropout(x)
        x = self.transformer_encoder(x)
        
        x = x[:, 0]
        x = self.classifier(x)
        return x