import torch
import torch.nn as nn

class FusionLayer(nn.Module):
    """
    Advanced Cross-Attention Fusion Layer (v4.0).
    Replaces simple concatenation with a Transformer-based multihead attention block
    so the speech and environmental representations can dynamically attend to each other.
    """
    def __init__(self, whisper_dim=512, clap_dim=512, fused_dim=256, num_heads=8):
        super().__init__()
        # Ensure dimensions match for attention
        self.w_proj = nn.Linear(whisper_dim, 512)
        self.c_proj = nn.Linear(clap_dim, 512)
        
        # Cross-modal self-attention
        self.attention = nn.MultiheadAttention(embed_dim=512, num_heads=num_heads, dropout=0.3, batch_first=True)
        self.layer_norm1 = nn.LayerNorm(512)
        self.layer_norm2 = nn.LayerNorm(512)
        
        # Feed Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(512, 1024),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.Dropout(0.3)
        )
        
        # Context Importance Gating
        self.context_gate = nn.Sequential(
            nn.Linear(512 * 2, 512 * 2),
            nn.Sigmoid()
        )
        
        # Final projection to fusion dim
        self.final_proj = nn.Sequential(
            nn.Linear(512 * 2, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.GELU()
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, whisper_emb, clap_emb):
        # whisper_emb: [B, 512] or [512]
        # clap_emb: [B, 512] or [512]
        if whisper_emb.dim() == 1:
            whisper_emb = whisper_emb.unsqueeze(0)
            clap_emb = clap_emb.unsqueeze(0)
            
        w = self.w_proj(whisper_emb) # [B, 512]
        c = self.c_proj(clap_emb)    # [B, 512]
        
        # Create sequence: [B, 2, 512] where token0=whisper, token1=clap
        seq = torch.stack([w, c], dim=1) 
        
        # Self-Attention
        attn_out, attn_weights = self.attention(seq, seq, seq)
        seq = self.layer_norm1(seq + attn_out)
        
        # FFN
        ffn_out = self.ffn(seq)
        seq = self.layer_norm2(seq + ffn_out)
        
        # Flatten back to [B, 1024]
        flattened = seq.view(seq.size(0), -1)
        
        # Adaptive Context Gating
        gate = self.context_gate(flattened)
        gated_features = flattened * gate
        
        # Project to fused dim
        fused = self.final_proj(gated_features)
        return fused # [B, fused_dim]