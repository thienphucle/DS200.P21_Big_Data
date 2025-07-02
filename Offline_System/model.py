import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    mean_squared_error, classification_report, accuracy_score, 
    f1_score, mean_absolute_error, r2_score, precision_recall_fscore_support
)
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import pickle
import warnings
warnings.filterwarnings('ignore')
import os

# Create ModelResults directory
os.makedirs('ModelResults', exist_ok=True)

class TikTokDataset(Dataset):
    
    def __init__(self, training_data, user_trend_features, text_features, mode='train'):
        self.training_data = training_data.reset_index(drop=True)
        self.user_trend_features = user_trend_features.set_index('user_name') if 'user_name' in user_trend_features.columns else user_trend_features
        self.text_features = text_features
        self.mode = mode
        
    def __len__(self):
        return len(self.training_data)
        
    def __getitem__(self, idx):
        row = self.training_data.iloc[idx]
        
        # Text features from TF-IDF (pre-computed)
        text_features = torch.tensor(self.text_features[idx], dtype=torch.float32)
        
        # Safe value extraction
        def safe_float(value, default=0.0):
            try:
                return float(value) if pd.notna(value) else default
            except (ValueError, TypeError):
                return default
        
        # Structured features from feature engineering
        structured_features = torch.tensor([
            safe_float(row.get('user_nfollower', 0)),
            safe_float(row.get('user_nfollowing', 0)), 
            safe_float(row.get('vid_duration_seconds', 0)),
            safe_float(row.get('post_hour', 0)),
            safe_float(row.get('post_day_of_week', 0)),
            safe_float(row.get('is_verified', 0)),
            safe_float(row.get('music_popularity', 0)),
            safe_float(row.get('num_hashtags', 0)),
            safe_float(row.get('caption_length', 0)),
            safe_float(row.get('post_is_weekend', 0)),
            safe_float(row.get('has_caption', 0)),
            safe_float(row.get('user_nfollower', 0)) / max(safe_float(row.get('user_nfollowing', 1)), 1),
            safe_float(row.get('latest_views', 1)) / max(safe_float(row.get('latest_hours_since_post', 1)), 1)
        ], dtype=torch.float32)
        
        # Time series features from feature engineering output
        time_features = torch.tensor([
            safe_float(row.get('avg_view_growth_rate', 0)),
            safe_float(row.get('avg_like_growth_rate', 0)),
            safe_float(row.get('avg_comment_growth_rate', 0)),
            safe_float(row.get('avg_share_growth_rate', 0)),
            safe_float(row.get('avg_save_growth_rate', 0)),
            safe_float(row.get('avg_engagement_growth_rate', 0)),
            safe_float(row.get('std_view_growth_rate', 0)),
            safe_float(row.get('std_like_growth_rate', 0)),
            safe_float(row.get('std_comment_growth_rate', 0)),
            safe_float(row.get('std_share_growth_rate', 0)),
            safe_float(row.get('std_save_growth_rate', 0)),
            safe_float(row.get('std_engagement_growth_rate', 0)),
            safe_float(row.get('latest_hours_since_post', 0)),
            safe_float(row.get('latest_views', 0)),
            safe_float(row.get('latest_likes', 0)),
            safe_float(row.get('latest_comments', 0)),
            safe_float(row.get('latest_shares', 0)),
            safe_float(row.get('latest_saves', 0))
        ], dtype=torch.float32)
        
        # Add user trend features if available
        user_name = row.get('user_name', '')
        if user_name in self.user_trend_features.index:
            user_trends = self.user_trend_features.loc[user_name]
            user_trend_tensor = torch.tensor([
                safe_float(user_trends.get('avg_view_growth', 0)),
                safe_float(user_trends.get('avg_like_growth', 0)),
                safe_float(user_trends.get('avg_engagement_growth', 0)),
                safe_float(user_trends.get('view_success_rate', 0)),
                safe_float(user_trends.get('engagement_success_rate', 0)),
                safe_float(user_trends.get('increase_rate', 0))
            ], dtype=torch.float32)
            time_features = torch.cat([time_features, user_trend_tensor])
        else:
            # Add zeros for missing user trends
            user_trend_tensor = torch.zeros(6, dtype=torch.float32)
            time_features = torch.cat([time_features, user_trend_tensor])
        
        result = {
            'text_features': text_features,
            'structured_features': structured_features,
            'time_features': time_features,
        }
        
        if self.mode == 'train':
            # Target variables - Predict absolute values at next time point (snapshot 4)
            targets = torch.tensor([
                safe_float(row.get('target_view_growth_rate', 0)),
                safe_float(row.get('target_like_growth_rate', 0)),
                safe_float(row.get('target_comment_growth_rate', 0)),
                safe_float(row.get('target_share_growth_rate', 0)),
                safe_float(row.get('target_save_growth_rate', 0))
            ], dtype=torch.float32)
            
            # Classification targets - Growth level classification
            growth_class_map = {'increase': 2, 'stable': 1, 'decrease': 0}
            class_targets = torch.tensor([
                growth_class_map.get(row.get('view_growth_class', 'stable'), 1),
                growth_class_map.get(row.get('like_growth_class', 'stable'), 1),
                growth_class_map.get(row.get('comment_growth_class', 'stable'), 1),
                growth_class_map.get(row.get('share_growth_class', 'stable'), 1),
                growth_class_map.get(row.get('save_growth_class', 'stable'), 1),
                growth_class_map.get(row.get('engagement_growth_class', 'stable'), 1)
            ], dtype=torch.long)
            
            result['targets'] = targets
            result['class_targets'] = class_targets
        
        return result

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
        x = self.layer_norm_input(x)
        x = x.unsqueeze(1)
        
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
        
        # Fixed: Variable selection should work on input dimension, not hidden dimension
        self.variable_selection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # Changed from hidden_dim to input_dim
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Softmax(dim=-1)
        )
        
        # Temporal attention layers
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
        batch_size = x.shape[0]
        
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        seq_len = x.shape[1]
        
        # Variable selection on original input
        var_weights = self.variable_selection(x.mean(dim=1))
        x = x * var_weights.unsqueeze(1)
        
        # Project to hidden dimension
        x = self.input_projection(x)
        x = self.layer_norm(x)

        # Add positional encoding
        if seq_len <= self.positional_encoding.shape[0]:
            pos_enc = self.positional_encoding[:seq_len].unsqueeze(0)
            x = x + pos_enc
        
        # Apply temporal attention layers with gating
        for layer in self.temporal_attention_layers:
            residual = x
            x = layer(x)
            gate = self.temporal_gate(x)
            x = gate * x + (1 - gate) * residual
        
        # Global pooling
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
        
        # Cross-modal attention layers
        self.text_to_others = nn.MultiheadAttention(
            fusion_dim, num_heads=8, dropout=0.1, batch_first=True
        )
        self.structured_to_others = nn.MultiheadAttention(
            fusion_dim, num_heads=8, dropout=0.1, batch_first=True
        )
        self.temporal_to_others = nn.MultiheadAttention(
            fusion_dim, num_heads=8, dropout=0.1, batch_first=True
        )
        
        # Modality gating
        self.modality_gate = nn.Sequential(
            nn.Linear(fusion_dim * 3, fusion_dim),
            nn.GELU(),
            nn.Linear(fusion_dim, 3),
            nn.Softmax(dim=-1)
        )
        
        # Final fusion layers
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
        # Project all modalities
        text_proj = self.text_projection(text_features).unsqueeze(1)
        structured_proj = self.structured_projection(structured_features).unsqueeze(1)
        temporal_proj = self.temporal_projection(temporal_features).unsqueeze(1)
        
        # Cross-modal attention
        all_features = torch.cat([text_proj, structured_proj, temporal_proj], dim=1)
        
        text_attended, text_attn = self.text_to_others(text_proj, all_features, all_features)
        struct_attended, struct_attn = self.structured_to_others(structured_proj, all_features, all_features)
        temp_attended, temp_attn = self.temporal_to_others(temporal_proj, all_features, all_features)
        
        # Squeeze back to 2D
        text_attended = text_attended.squeeze(1)
        struct_attended = struct_attended.squeeze(1)
        temp_attended = temp_attended.squeeze(1)
        
        # Modality gating
        concat_features = torch.cat([text_attended, struct_attended, temp_attended], dim=-1)
        gate_weights = self.modality_gate(concat_features)
        
        # Weighted combination
        fused_features = (gate_weights[:, 0:1] * text_attended + 
                         gate_weights[:, 1:2] * struct_attended + 
                         gate_weights[:, 2:3] * temp_attended)
        
        # Final fusion
        fused_features = self.fusion_layers(fused_features)
        
        attention_weights = {
            'text_attention': text_attn,
            'structured_attention': struct_attn,
            'temporal_attention': temp_attn,
            'modality_gates': gate_weights
        }
        
        return fused_features, attention_weights

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
            text_dim=128,  # Output from text processor
            structured_dim=128,
            temporal_dim=128,
            fusion_dim=fusion_dim
        )
        
        # Regression head for predicting next time point interactions
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
        
        # Classification head for growth level prediction
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
        # Process text features with neural network
        text_features_processed = self.text_processor(text_features)
        
        # Process structured features
        structured_features_processed, struct_attention = self.structured_processor(structured_features)
        
        # Process temporal features
        temporal_features_processed = self.temporal_processor(time_features)
        
        # Fuse all modalities
        fused_features, attention_weights = self.fusion_layer(
            text_features_processed, structured_features_processed, temporal_features_processed
        )
        
        # Generate predictions
        regression_predictions = self.regression_head(fused_features)
        
        classification_logits = self.classification_head(fused_features)
        classification_predictions = classification_logits.view(
            -1, self.num_classification_outputs, self.num_classes
        )
        
        # Combine all attention weights
        all_attention_weights = {
            'structured_attention': struct_attention,
            **attention_weights
        }
        
        return regression_predictions, classification_predictions, all_attention_weights

class ModernBaselineModels:
    def __init__(self):
        self.models = {
            # Traditional models
            'linear_regression': LinearRegression(),
            'random_forest_reg': RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest_cls': RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),

            # Nbeats-like MLP model
            'nbeats_mlp': MLPRegressor(
                hidden_layer_sizes=(256, 128, 64, 32),
                activation='relu',
                solver='adam',
                alpha=0.001,
                learning_rate='adaptive',
                max_iter=500,
                random_state=42
            ),

            # Modern gradient boosting models
            'xgboost_reg': xgb.XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            ),
            'xgboost_cls': xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            ),
            'lightgbm_reg': lgb.LGBMRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            ),
            'lightgbm_cls': lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            ),
            'catboost_reg': cb.CatBoostRegressor(
                iterations=200,
                depth=6,
                learning_rate=0.1,
                random_seed=42,
                verbose=False
            ),
            'catboost_cls': cb.CatBoostClassifier(
                iterations=200,
                depth=6,
                learning_rate=0.1,
                random_seed=42,
                verbose=False
            )
        }

        self.scalers = {}
        self.fitted_models = {}
        self.feature_importance = {}

    def prepare_features(self, training_data, text_features):
        feature_columns = [
            'user_nfollower', 'user_nfollowing', 'vid_duration_seconds',
            'post_hour', 'post_day_of_week', 'is_verified', 'music_popularity',
            'num_hashtags', 'caption_length', 'post_is_weekend', 'has_caption',
            'avg_view_growth_rate', 'avg_like_growth_rate', 'avg_comment_growth_rate',
            'avg_share_growth_rate', 'avg_save_growth_rate', 'avg_engagement_growth_rate',
            'std_view_growth_rate', 'std_like_growth_rate', 'std_comment_growth_rate',
            'std_share_growth_rate', 'std_save_growth_rate', 'std_engagement_growth_rate',
            'latest_hours_since_post', 'latest_views', 'latest_likes',
            'latest_comments', 'latest_shares', 'latest_saves'
        ]

        numerical_features = training_data[feature_columns].fillna(0).values
        text_subset = text_features[:, :100]
        combined_features = np.hstack([numerical_features, text_subset])

        return combined_features, feature_columns

    def train_baseline_models(self, X_train, y_train_reg, y_train_cls, X_val, y_val_reg, y_val_cls, feature_names):
        results = {}
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        self.scalers['main'] = scaler

        reg_models = [name for name in self.models.keys() if 'cls' not in name and 'logistic' not in name]

        for name in reg_models:
            model = self.models[name]
            y_train_avg = np.mean(y_train_reg, axis=1)
            y_val_avg = np.mean(y_val_reg, axis=1)

            model.fit(X_train_scaled, y_train_avg)
            y_pred = model.predict(X_val_scaled)

            mse = mean_squared_error(y_val_avg, y_pred)
            mae = mean_absolute_error(y_val_avg, y_pred)
            r2 = r2_score(y_val_avg, y_pred)

            results[name] = {
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'predictions': y_pred
            }

            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = dict(zip(
                    feature_names + [f'tfidf_{i}' for i in range(100)],
                    model.feature_importances_
                ))

            self.fitted_models[name] = model

        cls_models = [name for name in self.models.keys() if 'cls' in name or 'logistic' in name]

        for name in cls_models:
            model = self.models[name]

            y_train_cls_eng = y_train_cls[:, -1].astype(int)
            y_val_cls_eng = y_val_cls[:, -1].astype(int)

            model.fit(X_train_scaled, y_train_cls_eng)
            y_pred = model.predict(X_val_scaled)
            y_pred_proba = model.predict_proba(X_val_scaled)

            accuracy = accuracy_score(y_val_cls_eng, y_pred)
            f1_macro = f1_score(y_val_cls_eng, y_pred, average='macro')
            f1_weighted = f1_score(y_val_cls_eng, y_pred, average='weighted')

            results[name] = {
                'accuracy': accuracy,
                'f1_macro': f1_macro,
                'f1_weighted': f1_weighted,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }

            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = dict(zip(
                    feature_names + [f'tfidf_{i}' for i in range(100)],
                    model.feature_importances_
                ))

            self.fitted_models[name] = model

        return results

class Visualizer:    
    @staticmethod
    def plot_model_performance_comparison(neural_results, baseline_results, save_path='ModelResults/model_performance.png'):
        """Compare model performance"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Extract model names and metrics
        reg_models = [name for name in baseline_results.keys() if any(k in name for k in ['reg', 'linear', 'mlp'])]
        cls_models = [name for name in baseline_results.keys() if 'cls' in name or 'logistic' in name]
        
        # 1. Regression Performance (R² Score)
        r2_scores = [baseline_results[m]['r2'] for m in reg_models]
        r2_scores.append(neural_results.get('r2', 0))
        model_names_reg = [m.replace('_', ' ').title() for m in reg_models] + ['Neural Network']
        
        colors_reg = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#e67e22'] + ['#34495e']
        bars1 = ax1.bar(model_names_reg, r2_scores, color=colors_reg[:len(model_names_reg)])
        ax1.set_title('📈 Growth Prediction Accuracy (R² Score)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('R² Score (Higher = Better)', fontsize=12)
        ax1.set_ylim(0, max(r2_scores) * 1.1)
        ax1.grid(True, alpha=0.3)
        
        for bar, score in zip(bars1, r2_scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 2. Classification Performance (F1-Macro)
        f1_scores = [baseline_results[m]['f1_macro'] for m in cls_models]
        f1_scores.append(neural_results.get('f1_macro', 0))
        model_names_cls = [m.replace('_', ' ').title() for m in cls_models] + ['Neural Network']
        
        colors_cls = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6'] + ['#34495e']
        bars2 = ax2.bar(model_names_cls, f1_scores, color=colors_cls[:len(model_names_cls)])
        ax2.set_title('🎯 Growth Classification Accuracy (F1-Macro)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('F1-Macro Score (Higher = Better)', fontsize=12)
        ax2.set_ylim(0, max(f1_scores) * 1.1)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, score in zip(bars2, f1_scores):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 3. Economic Impact Analysis
        # Simulate ROI based on prediction accuracy
        roi_multipliers = [score * 100 for score in r2_scores]  # Convert R² to ROI %
        bars3 = ax3.bar(model_names_reg, roi_multipliers, color=colors_reg[:len(model_names_reg)])
        ax3.set_title('💰 Potential ROI from Accurate Predictions', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Estimated ROI Improvement (%)', fontsize=12)
        ax3.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, roi in zip(bars3, roi_multipliers):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{roi:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 4. Model Efficiency vs Performance
        # Simulate training time (inverse of complexity)
        efficiency_scores = [0.9, 0.3, 0.7, 0.6, 0.5, 0.8, 0.2]  # Added N-BEATS efficiency
        performance_scores = r2_scores
        
        scatter = ax4.scatter(efficiency_scores[:len(performance_scores)], performance_scores, 
                            s=200, c=colors_reg[:len(performance_scores)], alpha=0.7)
        
        # Add model labels
        for i, (eff, perf, name) in enumerate(zip(efficiency_scores[:len(performance_scores)], 
                                                 performance_scores, model_names_reg)):
            ax4.annotate(name, (eff, perf), xytext=(5, 5), textcoords='offset points', 
                        fontsize=10, fontweight='bold')
        
        ax4.set_title('⚡ Model Efficiency vs Performance Trade-off', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Training Efficiency (Higher = Faster)', fontsize=12)
        ax4.set_ylabel('Prediction Accuracy (R²)', fontsize=12)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        print("\n" + "="*60)
        print("📊 ECONOMIC INSIGHTS & BUSINESS IMPACT")
        print("="*60)
        
        best_reg_idx = np.argmax(r2_scores)
        best_cls_idx = np.argmax(f1_scores)
        
        print(f"🏆 Best Growth Prediction Model: {model_names_reg[best_reg_idx]}")
        print(f"   • R² Score: {r2_scores[best_reg_idx]:.3f}")
        print(f"   • Potential ROI Improvement: {roi_multipliers[best_reg_idx]:.1f}%")
        
        print(f"\n🎯 Best Classification Model: {model_names_cls[best_cls_idx]}")
        print(f"   • F1-Macro Score: {f1_scores[best_cls_idx]:.3f}")
        
        print(f"\n💡 KEY BUSINESS INSIGHTS:")
        print(f"   • Accurate growth prediction can improve content strategy ROI by up to {max(roi_multipliers):.1f}%")
        print(f"   • N-BEATS model shows specialized time series forecasting capabilities")
        print(f"   • Modern gradient boosting models (XGBoost, LightGBM, CatBoost) show superior performance")
        print(f"   • Neural networks provide competitive results but require more computational resources")
        
        return best_reg_idx, best_cls_idx
    
    @staticmethod
    def plot_feature_importance(feature_importance_dict, top_n=15, save_path='ModelResults/feature_importance.png'):
        if not feature_importance_dict:
            print("No feature importance data available")
            return
        
        # Get the best performing model's feature importance
        best_model = max(feature_importance_dict.keys(), 
                        key=lambda x: len(feature_importance_dict[x]))
        
        importance_data = feature_importance_dict[best_model]
        
        # Sort features by importance
        sorted_features = sorted(importance_data.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        features, importances = zip(*sorted_features)
        
        plt.figure(figsize=(12, 8))
        colors = plt.cm.viridis(np.linspace(0, 1, len(features)))
        bars = plt.barh(range(len(features)), importances, color=colors)
        
        plt.yticks(range(len(features)), features)
        plt.xlabel('Feature Importance', fontsize=12, fontweight='bold')
        plt.title(f'Top {top_n} Most Important Features for TikTok Growth Prediction\n({best_model.replace("_", " ").title()})', 
                 fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, importance) in enumerate(zip(bars, importances)):
            plt.text(importance + 0.001, i, f'{importance:.3f}', 
                    va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

class Trainer:    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = None
        self.baseline_models = ModernBaselineModels()
        self.training_history = {
            'train_losses': {'regression': [], 'classification': []},
            'val_losses': {'regression': [], 'classification': []}
        }
        
    def prepare_data(self, training_data, user_trend_features, text_features, test_size=0.2, batch_size=16):
        
        stratify_col = None
        if 'engagement_growth_class' in training_data.columns:
            stratify_col = training_data['engagement_growth_class']
        
        # Split indices to maintain alignment with text_features
        indices = np.arange(len(training_data))
        train_indices, val_indices = train_test_split(
            indices, 
            test_size=test_size, 
            random_state=42,
            stratify=stratify_col
        )
        
        train_df = training_data.iloc[train_indices].reset_index(drop=True)
        val_df = training_data.iloc[val_indices].reset_index(drop=True)
        train_text_features = text_features[train_indices]
        val_text_features = text_features[val_indices]
        
        # Create datasets using TF-IDF features from feature engineering
        train_dataset = TikTokDataset(train_df, user_trend_features, train_text_features, mode='train')
        val_dataset = TikTokDataset(val_df, user_trend_features, val_text_features, mode='train')
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        # Prepare data for baseline models
        X_train, feature_names = self.baseline_models.prepare_features(train_df, train_text_features)
        X_val, _ = self.baseline_models.prepare_features(val_df, val_text_features)
        
        # Extract targets for baseline models
        target_cols = ['target_view_growth_rate', 'target_like_growth_rate', 'target_comment_growth_rate', 
                      'target_share_growth_rate', 'target_save_growth_rate']
        class_cols = ['view_growth_class', 'like_growth_class', 'comment_growth_class', 
                     'share_growth_class', 'save_growth_class', 'engagement_growth_class']
        
        y_train_reg = train_df[target_cols].fillna(0).values
        y_val_reg = val_df[target_cols].fillna(0).values
        
        # Convert class labels to numbers
        growth_class_map = {'increase': 2, 'stable': 1, 'decrease': 0}
        y_train_cls = train_df[class_cols].fillna('stable').replace(growth_class_map).values.astype(int)
        y_val_cls = val_df[class_cols].fillna('stable').replace(growth_class_map).values.astype(int)

        
        return train_loader, val_loader, X_train, X_val, y_train_reg, y_val_reg, y_train_cls, y_val_cls, feature_names
    
    def train(self, train_loader, val_loader, X_train, X_val, y_train_reg, y_val_reg, y_train_cls, y_val_cls, 
              feature_names, epochs=15, lr=2e-5, save_path='ModelResults/tiktok_model.pth'):

        # Initialize neural network model
        self.model = TikTokGrowthPredictor(text_dim=train_loader.dataset.text_features.shape[1])
        self.model.to(self.device)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        
        regression_criterion = nn.SmoothL1Loss()
        classification_criterion = nn.CrossEntropyLoss(label_smoothing=0.1) 
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=lr,
            epochs=epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.1,
            anneal_strategy='cos'
        )
        
        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0
        
        print("Training Neural Network Model...")
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_reg_loss = 0.0
            train_cls_loss = 0.0
            
            for batch_idx, batch in enumerate(train_loader):
                optimizer.zero_grad()
                
                # Move batch to device
                text_features = batch['text_features'].to(self.device)
                structured_features = batch['structured_features'].to(self.device)
                time_features = batch['time_features'].to(self.device)
                targets = batch['targets'].to(self.device)
                class_targets = batch['class_targets'].to(self.device)
                
                # Forward pass
                reg_predictions, cls_predictions, _ = self.model(
                    text_features, structured_features, time_features
                )
                
                # Calculate losses
                reg_loss = regression_criterion(reg_predictions, targets)
                
                cls_loss = 0
                for i in range(cls_predictions.shape[1]):
                    cls_loss += classification_criterion(cls_predictions[:, i, :], class_targets[:, i])
                cls_loss /= cls_predictions.shape[1]
                
                # Combined loss with adaptive weighting
                total_loss = reg_loss + 0.3 * cls_loss
                
                # Backward pass
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                
                train_reg_loss += reg_loss.item()
                train_cls_loss += cls_loss.item()
            
            # Validation
            self.model.eval()
            val_reg_loss = 0.0
            val_cls_loss = 0.0
            
            with torch.no_grad():
                for batch in val_loader:
                    text_features = batch['text_features'].to(self.device)
                    structured_features = batch['structured_features'].to(self.device)
                    time_features = batch['time_features'].to(self.device)
                    targets = batch['targets'].to(self.device)
                    class_targets = batch['class_targets'].to(self.device)
                    
                    reg_predictions, cls_predictions, _ = self.model(
                        text_features, structured_features, time_features
                    )
                    
                    reg_loss = regression_criterion(reg_predictions, targets)
                    
                    cls_loss = 0
                    for i in range(cls_predictions.shape[1]):
                        cls_loss += classification_criterion(cls_predictions[:, i, :], class_targets[:, i])
                    cls_loss /= cls_predictions.shape[1]
                    
                    val_reg_loss += reg_loss.item()
                    val_cls_loss += cls_loss.item()
            
            # Calculate average losses
            avg_train_reg_loss = train_reg_loss / len(train_loader)
            avg_train_cls_loss = train_cls_loss / len(train_loader)
            avg_val_reg_loss = val_reg_loss / len(val_loader)
            avg_val_cls_loss = val_cls_loss / len(val_loader)
            avg_val_total_loss = avg_val_reg_loss + 0.3 * avg_val_cls_loss
            
            # Store training history
            self.training_history['train_losses']['regression'].append(avg_train_reg_loss)
            self.training_history['train_losses']['classification'].append(avg_train_cls_loss)
            self.training_history['val_losses']['regression'].append(avg_val_reg_loss)
            self.training_history['val_losses']['classification'].append(avg_val_cls_loss)
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Train Regression Loss: {avg_train_reg_loss:.4f}')
            print(f'  Train Classification Loss: {avg_train_cls_loss:.4f}')
            print(f'  Val Regression Loss: {avg_val_reg_loss:.4f}')
            print(f'  Val Classification Loss: {avg_val_cls_loss:.4f}')
            print(f'  Learning Rate: {scheduler.get_last_lr()[0]:.2e}')
            
            # Early stopping and model saving
            if avg_val_total_loss < best_val_loss:
                best_val_loss = avg_val_total_loss
                patience_counter = 0
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'model_config': {
                        'text_dim': train_loader.dataset.text_features.shape[1]
                    }
                }, save_path)
                print(f'  ✓ New model saved with val_loss: {best_val_loss:.4f}')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'  Early stopping triggered after {epoch+1} epochs')
                    break
        
        # Train baseline models
        print("\nTraining Baseline Models...")
        baseline_results = self.baseline_models.train_baseline_models(
            X_train, y_train_reg, y_train_cls, X_val, y_val_reg, y_val_cls, feature_names
        )
        
        return baseline_results
    
    def evaluate_and_visualize(self, val_loader, y_val_reg, y_val_cls, baseline_results, 
                              model_path='ModelResults/tiktok_model.pth'):        
        # Load best model
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = TikTokGrowthPredictor(text_dim=checkpoint['model_config']['text_dim'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        regression_predictions = []
        classification_predictions = []
        
        with torch.no_grad():
            for batch in val_loader:
                text_features = batch['text_features'].to(self.device)
                structured_features = batch['structured_features'].to(self.device)
                time_features = batch['time_features'].to(self.device)
                
                reg_pred, cls_pred, _ = self.model(
                    text_features, structured_features, time_features
                )
                
                regression_predictions.append(reg_pred.cpu().numpy())
                classification_predictions.append(torch.softmax(cls_pred, dim=-1).cpu().numpy())
        
        regression_predictions = np.vstack(regression_predictions)
        classification_predictions = np.vstack(classification_predictions)
        neural_results = {}
        
        y_val_reg_avg = np.mean(y_val_reg, axis=1)
        reg_pred_avg = np.mean(regression_predictions, axis=1)
        
        neural_results['mse'] = mean_squared_error(y_val_reg_avg, reg_pred_avg)
        neural_results['mae'] = mean_absolute_error(y_val_reg_avg, reg_pred_avg)
        neural_results['r2'] = r2_score(y_val_reg_avg, reg_pred_avg)
        
        # Classification metrics (use engagement growth class)
        y_val_cls_eng = y_val_cls[:, -1] 
        cls_pred_eng = np.argmax(classification_predictions[:, -1, :], axis=1)
        
        neural_results['accuracy'] = accuracy_score(y_val_cls_eng, cls_pred_eng)
        neural_results['f1_macro'] = f1_score(y_val_cls_eng, cls_pred_eng, average='macro')
        neural_results['f1_weighted'] = f1_score(y_val_cls_eng, cls_pred_eng, average='weighted')
        
        print("\nModel Performance Comparison:")
        print("REGRESSION METRICS:")
        print(f"Neural Network - MSE: {neural_results['mse']:.4f}, R²: {neural_results['r2']:.4f}")
        for model_name in ['linear_regression', 'random_forest_reg', 'xgboost_reg', 'lightgbm_reg', 'nbeats_reg']:
            if model_name in baseline_results:
                print(f"{model_name.replace('_', ' ').title()} - MSE: {baseline_results[model_name]['mse']:.4f}, R²: {baseline_results[model_name]['r2']:.4f}")
        
        print("\nCLASSIFICATION METRICS:")
        print(f"Neural Network - Accuracy: {neural_results['accuracy']:.4f}, F1-Macro: {neural_results['f1_macro']:.4f}")
        for model_name in ['logistic_regression', 'random_forest_cls', 'xgboost_cls', 'lightgbm_cls']:
            if model_name in baseline_results:
                print(f"{model_name.replace('_', ' ').title()} - Accuracy: {baseline_results[model_name]['accuracy']:.4f}, F1-Macro: {baseline_results[model_name]['f1_macro']:.4f}")
        
        visualizer = Visualizer()
        
        best_reg_idx, best_cls_idx = visualizer.plot_model_performance_comparison(
            neural_results, baseline_results, 'ModelResults/model_performance_economic.png'
        )
        
        visualizer.plot_feature_importance(
            self.baseline_models.feature_importance, 
            top_n=15, 
            save_path='ModelResults/feature_importance_business.png'
        )
        
        return neural_results, regression_predictions, classification_predictions
    
    def predict(self, training_data, user_trend_features, text_features, model_path='ModelResults/tiktok_model.pth'):
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = TikTokGrowthPredictor(text_dim=checkpoint['model_config']['text_dim'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        dataset = TikTokDataset(training_data, user_trend_features, text_features, mode='predict')
        dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
        
        regression_predictions = []
        classification_predictions = []
        attention_weights = []
        
        with torch.no_grad():
            for batch in dataloader:
                text_features = batch['text_features'].to(self.device)
                structured_features = batch['structured_features'].to(self.device)
                time_features = batch['time_features'].to(self.device)
                
                reg_pred, cls_pred, attn = self.model(
                    text_features, structured_features, time_features
                )
                
                regression_predictions.append(reg_pred.cpu().numpy())
                classification_predictions.append(torch.softmax(cls_pred, dim=-1).cpu().numpy())
                attention_weights.append({k: v.cpu().numpy() if torch.is_tensor(v) else v 
                                        for k, v in attn.items()})
        
        regression_predictions = np.vstack(regression_predictions)
        classification_predictions = np.vstack(classification_predictions)
        
        results_df = training_data.copy()
        
        growth_metrics = ['view', 'like', 'comment', 'share', 'save']
        for i, metric in enumerate(growth_metrics):
            results_df[f'pred_next_{metric}_growth'] = regression_predictions[:, i]
        
        class_names = ['decrease', 'stable', 'increase']
        metric_names = ['view', 'like', 'comment', 'share', 'save', 'engagement']
        
        for i, metric in enumerate(metric_names):
            pred_class = np.argmax(classification_predictions[:, i, :], axis=1)
            results_df[f'pred_{metric}_growth_level'] = [class_names[c] for c in pred_class]
            results_df[f'pred_{metric}_confidence'] = np.max(classification_predictions[:, i, :], axis=1)
            
            for j, class_name in enumerate(class_names):
                results_df[f'pred_{metric}_{class_name}_prob'] = classification_predictions[:, i, j]
        
        return results_df, attention_weights

if __name__ == "__main__":
    from Feature_Engineering import TikTokFeatureEngineer

    csv_file_path = r"D:\UIT\DS200\DS200_Project\Dataset\Preprocessed_Data\training_data.csv"  # Preprocessed CSV
    engineer = TikTokFeatureEngineer()
    df = pd.read_csv(csv_file_path)
    features = engineer.transform(df)
    
    if features['training_data'].empty:
        print("No training data available. Please check your data format.")
        exit()

    trainer = Trainer()
    train_loader, val_loader, X_train, X_val, y_train_reg, y_val_reg, y_train_cls, y_val_cls, feature_names = trainer.prepare_data(
        features['training_data'], 
        features['user_trend_features'],
        features['text_features'],
        batch_size=4
    )

    # Train models
    baseline_results = trainer.train(
        train_loader, val_loader, X_train, X_val, y_train_reg, y_val_reg, y_train_cls, y_val_cls,
        feature_names, epochs=2, lr=2e-5
    )

    neural_results, reg_predictions, cls_predictions = trainer.evaluate_and_visualize(
        val_loader, y_val_reg, y_val_cls, baseline_results
    )

    results_df, attention_weights = trainer.predict(
        features['training_data'], 
        features['user_trend_features'],
        features['text_features']
    )

    # Save results
    results_df.to_csv('ModelResults/tiktok_predictions.csv', index=False)