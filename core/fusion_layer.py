import torch
import torch.nn as nn

class FusionLayer(nn.Module):
    """
    ALM v7.0 Fusion Layer.
    Concatenates Speech (Whisper, 512) and Environmental (CLAP, 512) embeddings 
    and projects them into a Joint Representation (256) via an MLP.
    Architecture: [1024 -> 512 -> 256]
    """
    def __init__(self, whisper_dim=512, clap_dim=512, fused_dim=256):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(whisper_dim + clap_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, fused_dim),
            nn.BatchNorm1d(fused_dim),
            nn.GELU(),
            nn.Dropout(0.3)
        )
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, whisper_emb, clap_emb):
        """
        Args:
            whisper_emb: Tensor of shape [B, 512] or [512]
            clap_emb: Tensor of shape [B, 512] or [512]
        Returns:
            fused: Tensor of shape [B, 256]
        """
        if whisper_emb.dim() == 1:
            whisper_emb = whisper_emb.unsqueeze(0)
            clap_emb = clap_emb.unsqueeze(0)
            
        # V7.0 Simple Concatenation
        combined = torch.cat([whisper_emb, clap_emb], dim=1) # [B, 1024]
        
        # MLP Projection
        fused = self.mlp(combined) # [B, 256]
        
        return fused