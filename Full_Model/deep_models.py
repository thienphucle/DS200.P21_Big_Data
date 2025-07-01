import torch
import torch.nn as nn
import torch.nn.functional as F

class TabTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_heads=8, num_layers=4, dropout=0.1):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.layer_norm_input = nn.LayerNorm(hidden_dim)
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True
            ) for _ in range(num_layers)
        ])
        self.feature_attention = nn.MultiheadAttention(
            hidden_dim, num_heads=4, dropout=dropout, batch_first=True
        )
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.input_projection(x)
        x = self.layer_norm_input(x).unsqueeze(1)

        for layer in self.transformer_layers:
            x = layer(x)

        attended_x, attention_weights = self.feature_attention(x, x, x)
        x = x + attended_x
        x = x.squeeze(1)
        x = self.dropout(x)
        x = self.output_projection(x)

        return x, attention_weights


class TemporalFusionTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_heads=8, num_layers=4, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1000, hidden_dim) * 0.1)

        self.variable_selection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Softmax(dim=-1)
        )

        self.temporal_attention_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True
            ) for _ in range(num_layers)
        ])

        self.temporal_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        batch_size, seq_len = x.shape[0], x.shape[1]
        var_weights = self.variable_selection(x.mean(dim=1))
        x = x * var_weights.unsqueeze(1)

        x = self.input_projection(x)
        x = self.layer_norm(x)

        if seq_len <= self.positional_encoding.shape[0]:
            pos_enc = self.positional_encoding[:seq_len].unsqueeze(0)
            x = x + pos_enc

        for layer in self.temporal_attention_layers:
            residual = x
            x = layer(x)
            gate = self.temporal_gate(x)
            x = gate * x + (1 - gate) * residual

        x = x.mean(dim=1)
        return self.output_projection(x)


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

        self.text_to_others = nn.MultiheadAttention(fusion_dim, 8, dropout=0.1, batch_first=True)
        self.structured_to_others = nn.MultiheadAttention(fusion_dim, 8, dropout=0.1, batch_first=True)
        self.temporal_to_others = nn.MultiheadAttention(fusion_dim, 8, dropout=0.1, batch_first=True)

        self.modality_gate = nn.Sequential(
            nn.Linear(fusion_dim * 3, fusion_dim),
            nn.GELU(),
            nn.Linear(fusion_dim, 3),
            nn.Softmax(dim=-1)
        )

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

        text_att, text_attn = self.text_to_others(text_proj, all_features, all_features)
        struct_att, struct_attn = self.structured_to_others(structured_proj, all_features, all_features)
        temp_att, temp_attn = self.temporal_to_others(temporal_proj, all_features, all_features)

        text_att = text_att.squeeze(1)
        struct_att = struct_att.squeeze(1)
        temp_att = temp_att.squeeze(1)

        concat = torch.cat([text_att, struct_att, temp_att], dim=-1)
        gate_weights = self.modality_gate(concat)

        fused = gate_weights[:, 0:1] * text_att + gate_weights[:, 1:2] * struct_att + gate_weights[:, 2:3] * temp_att
        fused = self.fusion_layers(fused)

        attn_weights = {
            'text_attention': text_attn,
            'structured_attention': struct_attn,
            'temporal_attention': temp_attn,
            'modality_gates': gate_weights
        }

        return fused, attn_weights


class TikTokGrowthPredictor(nn.Module):
    def __init__(self, 
                 text_dim=512, 
                 structured_dim=13,
                 temporal_dim=24,
                 fusion_dim=512,
                 num_regression_outputs=5,
                 num_classification_outputs=6,
                 num_classes=3):
        super().__init__()

        self.text_processor = nn.Sequential(
            nn.Linear(text_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        self.structured_processor = TabTransformer(
            input_dim=structured_dim,
            hidden_dim=128,
            num_heads=8,
            num_layers=4
        )

        self.temporal_processor = TemporalFusionTransformer(
            input_dim=temporal_dim,
            hidden_dim=128,
            num_heads=8,
            num_layers=4
        )

        self.fusion_layer = MultiModalFusion(
            text_dim=128,
            structured_dim=128,
            temporal_dim=128,
            fusion_dim=fusion_dim
        )

        self.regression_head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.LayerNorm(fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim // 2, fusion_dim // 4),
            nn.LayerNorm(fusion_dim // 4),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(fusion_dim // 4, fusion_dim // 8),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim // 8, num_regression_outputs)
        )

        self.classification_head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.LayerNorm(fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim // 2, fusion_dim // 4),
            nn.LayerNorm(fusion_dim // 4),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(fusion_dim // 4, num_classification_outputs * num_classes)
        )

        self.num_classification_outputs = num_classification_outputs
        self.num_classes = num_classes
        self._init_weights()

    def _init_weights(self):
        for module in [self.text_processor, self.regression_head, self.classification_head]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.zeros_(layer.bias)

    def forward(self, text_features, structured_features, time_features):
        text_out = self.text_processor(text_features)
        struct_out, struct_attn = self.structured_processor(structured_features)
        temp_out = self.temporal_processor(time_features)

        fused, attention = self.fusion_layer(text_out, struct_out, temp_out)

        regression_preds = self.regression_head(fused)
        classification_logits = self.classification_head(fused)
        classification_preds = classification_logits.view(
            -1, self.num_classification_outputs, self.num_classes
        )

        all_attn = {
            'structured_attention': struct_attn,
            **attention
        }

        return regression_preds, classification_preds, all_attn
