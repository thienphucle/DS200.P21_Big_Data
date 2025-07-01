import torch
import torch.nn as nn

class MultiModalFusion(nn.Module):    
    def __init__(self, text_dim, structured_dim, temporal_dim, fusion_dim=512):
        super().__init__()
        
        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        self.structured_projection = nn.Sequential(
            nn.Linear(structured_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        self.temporal_projection = nn.Sequential(
            nn.Linear(temporal_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Cross-modal attention layers
        self.text_to_others = nn.MultiheadAttention(fusion_dim, num_heads=8, dropout=0.1, batch_first=True)
        self.structured_to_others = nn.MultiheadAttention(fusion_dim, num_heads=8, dropout=0.1, batch_first=True)
        self.temporal_to_others = nn.MultiheadAttention(fusion_dim, num_heads=8, dropout=0.1, batch_first=True)
        
        # Modality gating
        self.modality_gate = nn.Sequential(
            nn.Linear(fusion_dim * 3, fusion_dim),
            nn.GELU(),
            nn.Linear(fusion_dim, 3),
            nn.Softmax(dim=-1)
        )
        
        # Final fusion
        self.fusion_layers = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, text_features, structured_features, temporal_features):
        text_proj = self.text_projection(text_features).unsqueeze(1)
        structured_proj = self.structured_projection(structured_features).unsqueeze(1)
        temporal_proj = self.temporal_projection(temporal_features).unsqueeze(1)

        all_features = torch.cat([text_proj, structured_proj, temporal_proj], dim=1)
        
        text_attended, text_attn = self.text_to_others(text_proj, all_features, all_features)
        struct_attended, struct_attn = self.structured_to_others(structured_proj, all_features, all_features)
        temp_attended, temp_attn = self.temporal_to_others(temporal_proj, all_features, all_features)
        
        # Squeeze
        text_attended = text_attended.squeeze(1)
        struct_attended = struct_attended.squeeze(1)
        temp_attended = temp_attended.squeeze(1)
        
        # Modality gating
        concat_features = torch.cat([text_attended, struct_attended, temp_attended], dim=-1)
        gate_weights = self.modality_gate(concat_features)
        
        fused = (
            gate_weights[:, 0:1] * text_attended +
            gate_weights[:, 1:2] * struct_attended +
            gate_weights[:, 2:3] * temp_attended
        )
        
        fused_out = self.fusion_layers(fused)

        attention_weights = {
            'text_attention': text_attn,
            'structured_attention': struct_attn,
            'temporal_attention': temp_attn,
            'modality_gates': gate_weights
        }
        
        return fused_out, attention_weights
