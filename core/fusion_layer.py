import torch
import torch.nn as nn

class FusionLayer(nn.Module):
    def __init__(self, whisper_dim=512, clap_dim=512, fused_dim=256):
        super().__init__()
        self.w_norm = nn.LayerNorm(whisper_dim)
        self.c_norm = nn.LayerNorm(clap_dim)
        self.net = nn.Sequential(
            nn.Linear(whisper_dim + clap_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, whisper_emb, clap_emb):
        w = self.w_norm(whisper_emb)
        c = self.c_norm(clap_emb)
        combined = torch.cat([w, c], dim=-1)
        return self.net(combined) # [B, 256]