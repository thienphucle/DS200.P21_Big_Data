import torch
from torch.utils.data import Dataset
import pandas as pd

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