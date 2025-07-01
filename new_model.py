import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, f1_score
)
import numpy as np
from sklearn.linear_model import LinearRegression, ElasticNet, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
from feature_engineering import TikTokFeatureEngineer

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("CatBoost not available. Install with: pip install catboost")

import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

### Dataset
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
            # Add follower/following ratio
            safe_float(row.get('user_nfollower', 0)) / max(safe_float(row.get('user_nfollowing', 1)), 1),
            # Add engagement rate from latest snapshot
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

### Baseline Models
class ModernBaselineModels:
    def __init__(self):
        self.models = {
            # Linear models
            'linear_regression': LinearRegression(),
            'elastic_net': ElasticNet(),
            'ridge': Ridge(),
            'lasso': Lasso(),

            # Tree-based models
            'random_forest_reg': RandomForestRegressor(n_estimators=200, random_state=42),
            'extra_trees_reg': ExtraTreesRegressor(n_estimators=200, random_state=42),
            'xgboost_reg': xgb.XGBRegressor(n_estimators=200, random_state=42),
            'lightgbm_reg': lgb.LGBMRegressor(n_estimators=200, random_state=42),

            # Classification models
            'logistic_regression': LogisticRegression(max_iter=1000),
            'random_forest_cls': RandomForestClassifier(n_estimators=200, random_state=42),
            'xgboost_cls': xgb.XGBClassifier(n_estimators=200, random_state=42),
            'lightgbm_cls': lgb.LGBMClassifier(n_estimators=200, random_state=42),
        }

        if CATBOOST_AVAILABLE:
            self.models.update({
                'catboost_reg': cb.CatBoostRegressor(iterations=200, verbose=False),
                'catboost_cls': cb.CatBoostClassifier(iterations=200, verbose=False),
            })

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
        numeric = training_data[feature_columns].fillna(0).values
        text_subset = text_features[:, :100]  # reduce TF-IDF size
        combined = np.hstack([numeric, text_subset])
        return combined, feature_columns

    def train_baseline_models(self, X_train, y_train_reg, y_train_cls, X_val, y_val_reg, y_val_cls, feature_names):
        results = {}
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        self.scalers['main'] = scaler

        # Regression Models
        reg_models = [name for name in self.models if name.endswith('_reg') or 'regression' in name]
        for name in reg_models:
            model = self.models[name]
            y_train_avg = np.mean(y_train_reg, axis=1)
            y_val_avg = np.mean(y_val_reg, axis=1)

            if 'linear' in name or name in ['elastic_net', 'ridge', 'lasso']:
                model.fit(X_train_scaled, y_train_avg)
                y_pred = model.predict(X_val_scaled)
            else:
                model.fit(X_train, y_train_avg)
                y_pred = model.predict(X_val)

            results[name] = {
                'mse': mean_squared_error(y_val_avg, y_pred),
                'mae': mean_absolute_error(y_val_avg, y_pred),
                'r2': r2_score(y_val_avg, y_pred),
                'predictions': y_pred
            }

            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = dict(zip(
                    feature_names + [f'tfidf_{i}' for i in range(100)],
                    model.feature_importances_
                ))

            self.fitted_models[name] = model

        # Classification Models
        cls_models = [name for name in self.models if name.endswith('_cls') or 'logistic' in name]
        for name in cls_models:
            model = self.models[name]
            y_train_cls_eng = y_train_cls[:, -1]
            y_val_cls_eng = y_val_cls[:, -1]

            if 'logistic' in name:
                model.fit(X_train_scaled, y_train_cls_eng)
                y_pred = model.predict(X_val_scaled)
            else:
                model.fit(X_train, y_train_cls_eng)
                y_pred = model.predict(X_val)

            results[name] = {
                'accuracy': accuracy_score(y_val_cls_eng, y_pred),
                'f1_macro': f1_score(y_val_cls_eng, y_pred, average='macro'),
                'f1_weighted': f1_score(y_val_cls_eng, y_pred, average='weighted'),
                'predictions': y_pred
            }

            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = dict(zip(
                    feature_names + [f'tfidf_{i}' for i in range(100)],
                    model.feature_importances_
                ))

            self.fitted_models[name] = model

        return results

### Deep Learning Model
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

### AutoML models
try:
    from flaml import AutoML
    FLAML_AVAILABLE = True
except ImportError:
    FLAML_AVAILABLE = False

try:
    import autosklearn.regression
    import autosklearn.classification
    AUTOSKLEARN_AVAILABLE = True
except ImportError:
    AUTOSKLEARN_AVAILABLE = False

try:
    from tpot import TPOTRegressor, TPOTClassifier
    TPOT_AVAILABLE = True
except ImportError:
    TPOT_AVAILABLE = False

try:
    import h2o
    from h2o.automl import H2OAutoML
    H2O_AVAILABLE = True
    h2o.init()
except:
    H2O_AVAILABLE = False

try:
    from pycaret.regression import setup as pycaret_reg_setup, compare_models as pycaret_compare_models_reg
    from pycaret.classification import setup as pycaret_cls_setup, compare_models as pycaret_compare_models_cls
    PYCARET_AVAILABLE = True
except:
    PYCARET_AVAILABLE = False


class AutoMLModels:
    def __init__(self):
        self.models = {}
        self.available = {
            'flaml': FLAML_AVAILABLE,
            'autosklearn': AUTOSKLEARN_AVAILABLE,
            'tpot': TPOT_AVAILABLE,
            'h2o': H2O_AVAILABLE,
            'pycaret': PYCARET_AVAILABLE
        }

    def train_flaml(self, X_train, y_train, task='regression', time_budget=30):
        if not FLAML_AVAILABLE:
            raise ImportError("FLAML not installed")
        automl = AutoML()
        automl.fit(X_train=X_train, y_train=y_train, task=task, time_budget=time_budget)
        return automl

    def train_autosklearn(self, X_train, y_train, task='regression'):
        if not AUTOSKLEARN_AVAILABLE:
            raise ImportError("AutoSklearn not installed")

        if task == 'regression':
            model = autosklearn.regression.AutoSklearnRegressor(time_left_for_this_task=60, per_run_time_limit=30)
        else:
            model = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=60, per_run_time_limit=30)
        model.fit(X_train, y_train)
        return model

    def train_tpot(self, X_train, y_train, task='regression'):
        if not TPOT_AVAILABLE:
            raise ImportError("TPOT not installed")

        if task == 'regression':
            model = TPOTRegressor(generations=5, population_size=20, verbosity=2)
        else:
            model = TPOTClassifier(generations=5, population_size=20, verbosity=2)
        model.fit(X_train, y_train)
        return model

    def train_h2o(self, X_train, y_train, task='regression'):
        if not H2O_AVAILABLE:
            raise ImportError("H2O not installed or initialized")

        import pandas as pd
        train_df = pd.DataFrame(X_train)
        train_df['target'] = y_train
        h2o_frame = h2o.H2OFrame(train_df)

        aml = H2OAutoML(max_runtime_secs=60, seed=1)
        if task == 'regression':
            aml.train(x=h2o_frame.columns[:-1], y='target', training_frame=h2o_frame)
        else:
            h2o_frame['target'] = h2o_frame['target'].asfactor()
            aml.train(x=h2o_frame.columns[:-1], y='target', training_frame=h2o_frame)

        return aml

    def train_pycaret(self, X_train, y_train, task='regression'):
        if not PYCARET_AVAILABLE:
            raise ImportError("PyCaret not installed")

        import pandas as pd
        data = pd.DataFrame(X_train)
        data['target'] = y_train

        if task == 'regression':
            s = pycaret_reg_setup(data=data, target='target', silent=True, verbose=False)
            model = pycaret_compare_models_reg()
        else:
            s = pycaret_cls_setup(data=data, target='target', silent=True, verbose=False)
            model = pycaret_compare_models_cls()

        return model

### Fusion Model
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

### Tabnet Model
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

try:
    from pytorch_tabnet.tab_model import TabNetRegressor, TabNetClassifier
    TABNET_AVAILABLE = True
except ImportError:
    TABNET_AVAILABLE = False
    print("❌ TabNet not installed. Run: pip install pytorch-tabnet")

class TabNetModel:
    def __init__(self):
        if not TABNET_AVAILABLE:
            raise ImportError("TabNet not available.")
        self.regressor = None
        self.classifier = None
        self.scaler = RobustScaler()

    def fit_regression(self, X, y, max_epochs=100):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)

        self.regressor = TabNetRegressor(verbose=0)
        self.regressor.fit(X_train, y_train, eval_set=[(X_val, y_val)], patience=10)

        preds = self.regressor.predict(X_val)
        mse = mean_squared_error(y_val, preds)
        return self.regressor, mse

    def fit_classification(self, X, y, max_epochs=100):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)

        self.classifier = TabNetClassifier(verbose=0)
        self.classifier.fit(X_train, y_train, eval_set=[(X_val, y_val)], patience=10)

        preds = self.classifier.predict(X_val)
        acc = accuracy_score(y_val, preds)
        f1 = f1_score(y_val, preds, average='macro')
        return self.classifier, acc, f1

### Evaluate
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

def evaluate_model(y_true_reg, y_pred_reg, y_true_cls, y_pred_cls):
    results = {}

    y_true_avg = np.mean(y_true_reg, axis=1)
    y_pred_avg = np.mean(y_pred_reg, axis=1)

    mae = mean_absolute_error(y_true_avg, y_pred_avg)
    rmse = mean_squared_error(y_true_avg, y_pred_avg, squared=False)
    r2 = r2_score(y_true_avg, y_pred_avg)
    mape = mean_absolute_percentage_error(y_true_avg, y_pred_avg)

    results['regression'] = {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'MAPE': mape
    }

    y_true_cls_eng = y_true_cls[:, -1]
    y_pred_cls_eng = y_pred_cls[:, -1]

    acc = accuracy_score(y_true_cls_eng, y_pred_cls_eng)
    f1 = f1_score(y_true_cls_eng, y_pred_cls_eng, average='macro')
    precision, recall, _, _ = precision_recall_fscore_support(
        y_true_cls_eng, y_pred_cls_eng, average=None, labels=[0, 1, 2]
    )
    conf_matrix = confusion_matrix(y_true_cls_eng, y_pred_cls_eng, labels=[0, 1, 2])

    results['classification'] = {
        'Accuracy': acc,
        'F1 Macro': f1,
        'Precision': precision.tolist(),
        'Recall': recall.tolist(),
        'Confusion Matrix': conf_matrix.tolist()
    }

    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Decrease', 'Stable', 'Increase'], yticklabels=['Decrease', 'Stable', 'Increase'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

    return results

def recommend_top_kols(df, top_k=10):
    if 'engagement_growth_class' not in df.columns or 'engagement_rate' not in df.columns:
        print("Missing necessary columns in dataframe.")
        return pd.DataFrame()

    filtered = df[
        (df['engagement_growth_class'] == 'increase') &
        (df['engagement_rate'] > df['engagement_rate'].median())
    ]

    top_kols = filtered.groupby('user_name').agg({
        'user_nfollower': 'first',
        'engagement_rate': 'mean',
        'topic': lambda x: x.mode()[0] if not x.mode().empty else None
    }).sort_values('engagement_rate', ascending=False).head(top_k)

    top_kols = top_kols.reset_index()
    top_kols.to_csv('top_kols.csv', index=False)
    print("✅ Top KOLs exported to top_kols.csv")

    return top_kols

### Trainer
from dataset import TikTokDataset
from classic_models import ModernBaselineModels
from deep_models import TikTokGrowthPredictor
from visualizer import Visualizer
import json

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
        stratify_col = training_data['engagement_growth_class'] if 'engagement_growth_class' in training_data.columns else None
        indices = np.arange(len(training_data))
        train_idx, val_idx = train_test_split(indices, test_size=test_size, random_state=42, stratify=stratify_col)
        
        train_df = training_data.iloc[train_idx].reset_index(drop=True)
        val_df = training_data.iloc[val_idx].reset_index(drop=True)
        train_text = text_features[train_idx]
        val_text = text_features[val_idx]

        train_ds = TikTokDataset(train_df, user_trend_features, train_text, mode='train')
        val_ds = TikTokDataset(val_df, user_trend_features, val_text, mode='train')
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        X_train, feature_names = self.baseline_models.prepare_features(train_df, train_text)
        X_val, _ = self.baseline_models.prepare_features(val_df, val_text)

        y_train_reg = train_df.filter(like='target_').values
        y_val_reg = val_df.filter(like='target_').values

        growth_map = {'increase': 2, 'stable': 1, 'decrease': 0}
        y_train_cls = train_df.filter(like='_class').replace(growth_map).values.astype(int)
        y_val_cls = val_df.filter(like='_class').replace(growth_map).values.astype(int)

        return train_loader, val_loader, X_train, X_val, y_train_reg, y_val_reg, y_train_cls, y_val_cls, feature_names
    
    def train(self, train_loader, val_loader, X_train, X_val, y_train_reg, y_val_reg, y_train_cls, y_val_cls,
              feature_names, epochs=15, lr=2e-5, save_path='best_tiktok_model.pth'):

        self.model = TikTokGrowthPredictor(text_dim=train_loader.dataset.text_features.shape[1])
        self.model.to(self.device)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        reg_loss_fn = nn.SmoothL1Loss()
        cls_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=len(train_loader)
        )
        
        best_val_loss = float('inf')
        patience = 5
        counter = 0
        
        for epoch in range(epochs):
            self.model.train()
            total_reg, total_cls = 0, 0
            
            for batch in train_loader:
                optimizer.zero_grad()
                text, struct, time = batch['text_features'].to(self.device), batch['structured_features'].to(self.device), batch['time_features'].to(self.device)
                y_reg, y_cls = batch['targets'].to(self.device), batch['class_targets'].to(self.device)

                out_reg, out_cls, _ = self.model(text, struct, time)
                loss_reg = reg_loss_fn(out_reg, y_reg)
                loss_cls = sum(cls_loss_fn(out_cls[:, i], y_cls[:, i]) for i in range(out_cls.shape[1])) / out_cls.shape[1]
                loss = loss_reg + 0.3 * loss_cls
                loss.backward()
                optimizer.step()
                scheduler.step()
                total_reg += loss_reg.item()
                total_cls += loss_cls.item()
            
            self.model.eval()
            val_reg, val_cls = 0, 0
            with torch.no_grad():
                for batch in val_loader:
                    text, struct, time = batch['text_features'].to(self.device), batch['structured_features'].to(self.device), batch['time_features'].to(self.device)
                    y_reg, y_cls = batch['targets'].to(self.device), batch['class_targets'].to(self.device)
                    out_reg, out_cls, _ = self.model(text, struct, time)
                    loss_reg = reg_loss_fn(out_reg, y_reg)
                    loss_cls = sum(cls_loss_fn(out_cls[:, i], y_cls[:, i]) for i in range(out_cls.shape[1])) / out_cls.shape[1]
                    val_reg += loss_reg.item()
                    val_cls += loss_cls.item()

            avg_val_loss = val_reg + 0.3 * val_cls
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                counter = 0
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'model_config': {'text_dim': train_loader.dataset.text_features.shape[1]}
                }, save_path)
            else:
                counter += 1
                if counter >= patience:
                    break
        
        print("Finished training neural model.")
        baseline_results = self.baseline_models.train_baseline_models(X_train, y_train_reg, y_train_cls, X_val, y_val_reg, y_val_cls, feature_names)
        return baseline_results
    
    def predict(self, training_data, user_trend_features, text_features, model_path='best_tiktok_model.pth'):
        self.model.eval()
        dataset = TikTokDataset(training_data, user_trend_features, text_features, mode='train')
        loader = DataLoader(dataset, batch_size=16, shuffle=False)

        reg_preds, cls_preds, attentions = [], [], []
        with torch.no_grad():
            for batch in loader:
                text, struct, time = batch['text_features'].to(self.device), batch['structured_features'].to(self.device), batch['time_features'].to(self.device)
                reg_out, cls_out, attn = self.model(text, struct, time)
                reg_preds.append(reg_out.cpu().numpy())
                cls_preds.append(torch.argmax(cls_out, dim=-1).cpu().numpy())
                attentions.append(attn)

        reg_preds = np.vstack(reg_preds)
        cls_preds = np.vstack(cls_preds)
        results_df = training_data.copy()
        results_df[[f'pred_target_{i}' for i in range(reg_preds.shape[1])]] = reg_preds
        results_df[[f'pred_class_{i}' for i in range(cls_preds.shape[1])]] = cls_preds
        return results_df, attentions

    
    def evaluate_and_visualize(self, val_loader, y_val_reg, y_val_cls, baseline_results, model_path='best_tiktok_model.pth'):
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = TikTokGrowthPredictor(text_dim=checkpoint['model_config']['text_dim'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        reg_preds, cls_preds = [], []
        with torch.no_grad():
            for batch in val_loader:
                text, struct, time = batch['text_features'].to(self.device), batch['structured_features'].to(self.device), batch['time_features'].to(self.device)
                reg_out, cls_out, _ = self.model(text, struct, time)
                reg_preds.append(reg_out.cpu().numpy())
                cls_preds.append(torch.softmax(cls_out, dim=-1).cpu().numpy())

        reg_preds = np.vstack(reg_preds)
        cls_preds = np.vstack(cls_preds)

        results_dict = evaluate_model(y_val_reg, reg_preds, y_val_cls, cls_preds.argmax(axis=-1))
        with open("evaluation_report.json", "w") as f:
            json.dump(results_dict, f, indent=2)
        print("✅ Evaluation report saved to evaluation_report.json")

        Visualizer.plot_model_performance_comparison(results_dict['regression'] | {'f1_macro': results_dict['classification']['F1 Macro']}, baseline_results)
        Visualizer.plot_feature_importance(self.baseline_models.feature_importance)
        return results_dict

### Visualizer
class Visualizer:
    
    @staticmethod
    def plot_model_performance_comparison(neural_results, baseline_results, save_path='model_performance.png'):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Regression comparison (R²)
        reg_models = [k for k in baseline_results if 'reg' in k or 'linear' in k]
        r2_scores = [baseline_results[m].get('r2', 0) for m in reg_models]
        r2_scores.append(neural_results.get('r2', 0))
        model_names = [m.replace('_', ' ').title() for m in reg_models] + ['NeuralNet']
        
        bars = ax1.bar(model_names, r2_scores, color='skyblue')
        ax1.set_title("Regression Performance (R² Score)")
        ax1.set_ylabel("R²")
        ax1.set_ylim(0, max(r2_scores) + 0.05)
        ax1.grid(True, linestyle='--', alpha=0.5)
        for bar, score in zip(bars, r2_scores):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{score:.3f}", ha='center', va='bottom')
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')

        # Classification comparison (F1 Macro)
        cls_models = [k for k in baseline_results if 'cls' in k or 'logistic' in k]
        f1_scores = [baseline_results[m].get('f1_macro', 0) for m in cls_models]
        f1_scores.append(neural_results.get('f1_macro', 0))
        model_names_cls = [m.replace('_', ' ').title() for m in cls_models] + ['NeuralNet']

        bars2 = ax2.bar(model_names_cls, f1_scores, color='lightgreen')
        ax2.set_title("Classification Performance (F1 Macro)")
        ax2.set_ylabel("F1 Macro")
        ax2.set_ylim(0, max(f1_scores) + 0.05)
        ax2.grid(True, linestyle='--', alpha=0.5)
        for bar, score in zip(bars2, f1_scores):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{score:.3f}", ha='center', va='bottom')
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.show()

    @staticmethod
    def plot_feature_importance(feature_importance_dict, top_n=15, save_path='feature_importance.png'):
        if not feature_importance_dict:
            print("No feature importance found.")
            return

        best_model = max(feature_importance_dict, key=lambda k: len(feature_importance_dict[k]))
        importances = feature_importance_dict[best_model]
        sorted_feat = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:top_n]

        features, scores = zip(*sorted_feat)
        plt.figure(figsize=(10, 6))
        bars = plt.barh(features, scores, color=plt.cm.viridis(np.linspace(0.2, 0.8, top_n)))
        plt.gca().invert_yaxis()
        plt.xlabel("Importance")
        plt.title(f"Top {top_n} Important Features ({best_model})")
        for bar, val in zip(bars, scores):
            plt.text(val + 0.01, bar.get_y() + bar.get_height()/2, f"{val:.3f}", va='center')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.show()

import argparse
import pandas as pd
import numpy as np
from trainer import Trainer
from evaluate import evaluate_model, recommend_top_kols
from visualizer import Visualizer
from automl_models import AutoMLModels
from tabnet_model import TabNetModel
from feature_engineering import TikTokFeatureEngineer

def main(args):
    # Load and transform data
    df = pd.read_csv(args.data_path)
    engineer = TikTokFeatureEngineer()
    features = engineer.transform(df)

    if features['training_data'].empty:
        print("❌ No valid training data found.")
        return

    trainer = Trainer()
    train_loader, val_loader, X_train, X_val, y_train_reg, y_val_reg, y_train_cls, y_val_cls, feature_names = trainer.prepare_data(
        features['training_data'],
        features['user_trend_features'],
        features['text_features'],
        batch_size=args.batch_size
    )

    # Train deep learning model
    print("🔁 Training deep learning model...")
    baseline_results = trainer.train(
        train_loader, val_loader, X_train, X_val,
        y_train_reg, y_val_reg, y_train_cls, y_val_cls,
        feature_names, epochs=args.epochs, lr=args.lr, save_path=args.save_path
    )

    neural_results = trainer.evaluate_and_visualize(
        val_loader, y_val_reg, y_val_cls, baseline_results,
        model_path=args.save_path
    )

    results_df, attn = trainer.predict(
        features['training_data'],
        features['user_trend_features'],
        features['text_features'],
        model_path=args.save_path
    )
    results_df.to_csv(args.output_path, index=False)
    print(f"✅ Predictions saved to {args.output_path}")

    top_kols = recommend_top_kols(results_df)
    print(top_kols)

    # Train TabNet model if available
    if TABNET_AVAILABLE:
        print("🔁 Training TabNet models...")
        tabnet = TabNetModel()
        tabnet_reg, tabnet_mse = tabnet.fit_regression(X_train, np.mean(y_train_reg, axis=1))
        tabnet_cls, tabnet_acc, tabnet_f1 = tabnet.fit_classification(X_train, y_train_cls[:, -1])

        print(f"✅ TabNet Regression MSE: {tabnet_mse:.4f}")
        print(f"✅ TabNet Classification Accuracy: {tabnet_acc:.4f}, F1 Macro: {tabnet_f1:.4f}")

    # Train AutoML models if available
    automl = AutoMLModels()
    if automl.available['flaml']:
        print("🔁 Training AutoML (FLAML) models...")
        flaml_reg = automl.train_flaml(X_train, np.mean(y_train_reg, axis=1), task='regression', time_budget=30)
        flaml_cls = automl.train_flaml(X_train, y_train_cls[:, -1], task='classification', time_budget=30)

        y_pred_flaml_reg = flaml_reg.predict(X_val)
        y_pred_flaml_cls = flaml_cls.predict(X_val)

        print(f"✅ FLAML Regression R2: {r2_score(np.mean(y_val_reg, axis=1), y_pred_flaml_reg):.4f}")
        print(f"✅ FLAML Classification F1 Macro: {f1_score(y_val_cls[:, -1], y_pred_flaml_cls, average='macro'):.4f}")

    print("✅ All models trained and evaluated.")


if __name__ == "__main__":
    main()

