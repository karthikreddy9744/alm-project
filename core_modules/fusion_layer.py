import torch
import torch.nn as nn

class FusionLayer(nn.Module):
    """
    ALM v10.7 Unified Fusion Layer.
    Concatenates Speech (Whisper, 512), Semantic Audio (CLAP, 512), 
    and Polyphonic Event Detection (HTS-AT, 768) embeddings 
    and projects them into a Joint Representation (256) via an MLP.
    Architecture: [1792 -> 512 -> 256]
    """
    def __init__(self, whisper_dim=512, clap_dim=512, htsat_dim=768, fused_dim=256):
        super().__init__()
        
        input_dim = whisper_dim + clap_dim + htsat_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 512),
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

    def forward(self, whisper_emb, clap_emb, htsat_emb):
        """
        Args:
            whisper_emb: Tensor of shape [B, 512] or [512]
            clap_emb: Tensor of shape [B, 512] or [512]
            htsat_emb: Tensor of shape [B, 768] or [768]
        Returns:
            fused: Tensor of shape [B, 256]
        """
        if whisper_emb.dim() == 1:
            whisper_emb = whisper_emb.unsqueeze(0)
            clap_emb = clap_emb.unsqueeze(0)
            htsat_emb = htsat_emb.unsqueeze(0)
            
        # Unified Concatenation
        combined = torch.cat([whisper_emb, clap_emb, htsat_emb], dim=1) # [B, 1792]
        
        # MLP Projection
        fused = self.mlp(combined) # [B, 256]
        
        return fused