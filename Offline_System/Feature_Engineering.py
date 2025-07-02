import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import re
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, RobustScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class TikTokFeatureEngineer:
    def __init__(self, n_recent_videos: int = 20, min_snapshots: int = 4, max_text_features: int = 512):
        self.n_recent = n_recent_videos
        self.min_snapshots = min_snapshots
        self.max_text_features = max_text_features
        self.feature_columns = []
        self.text_vectorizer = None
        self.scalers = {}
        
    def _parse_count(self, count_str: str) -> float:
        if pd.isna(count_str) or count_str == '':
            return 0.0
        
        count_str = str(count_str).replace(',', '').strip()
        
        try:
            if 'K' in count_str.upper():
                return float(count_str.upper().replace('K', '')) * 1000
            elif 'M' in count_str.upper():
                return float(count_str.upper().replace('M', '')) * 1000000
            elif 'B' in count_str.upper():
                return float(count_str.upper().replace('B', '')) * 1000000000
            else:
                return float(count_str)
        except (ValueError, TypeError):
            return 0.0
    
    def _extract_hashtags(self, hashtag_str) -> List[str]:
        if pd.isna(hashtag_str) or hashtag_str == '':
            return []
        
        if isinstance(hashtag_str, list):
            return hashtag_str
        
        hashtag_str = str(hashtag_str).replace('"', '').replace("'", '').replace('[', '').replace(']', '')
        hashtags = [tag.strip() for tag in hashtag_str.split(',')]
        return [tag for tag in hashtags if tag and not tag.isspace()]
    
    def _prepare_text_features(self, df: pd.DataFrame) -> np.ndarray:
        text_content = []
        for idx in range(len(df)):
            row = df.iloc[idx]
            caption = str(row.get('vid_caption', ''))
            hashtags = str(row.get('vid_hashtag', ''))
            category = str(row.get('vid_category', ''))
            
            combined_text = f"{caption} {hashtags} {category}".strip()
            if not combined_text or combined_text.isspace():
                combined_text = "no content"
            text_content.append(combined_text)
        
        if self.text_vectorizer is None:
            self.text_vectorizer = TfidfVectorizer(
                max_features=self.max_text_features,
                stop_words='english',
                ngram_range=(1, 3), 
                min_df=2,
                max_df=0.95,
                sublinear_tf=True 
            )
            text_features = self.text_vectorizer.fit_transform(text_content).toarray()
        else:
            text_features = self.text_vectorizer.transform(text_content).toarray()
        
        return text_features
    
    def _calculate_engagement_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Total engagement
        df['total_engagement'] = df['vid_nlike'] + df['vid_ncomment'] + df['vid_nshare'] + df['vid_nsave']
        
        # Engagement rates (avoid division by zero)
        df['like_rate'] = df['vid_nlike'] / np.maximum(df['vid_nview'], 1)
        df['comment_rate'] = df['vid_ncomment'] / np.maximum(df['vid_nview'], 1)
        df['share_rate'] = df['vid_nshare'] / np.maximum(df['vid_nview'], 1)
        df['save_rate'] = df['vid_nsave'] / np.maximum(df['vid_nview'], 1)
        df['engagement_rate'] = df['total_engagement'] / np.maximum(df['vid_nview'], 1)
        
        # Engagement quality score (weighted by action importance)
        df['engagement_quality'] = (
            df['vid_nlike'] * 1.0 + 
            df['vid_ncomment'] * 3.0 +  # Comments are more valuable
            df['vid_nshare'] * 5.0 +    # Shares are very valuable
            df['vid_nsave'] * 4.0       # Saves indicate strong interest
        ) / np.maximum(df['vid_nview'], 1)
        
        # Engagement velocity (engagement per view per hour)
        df['engagement_velocity'] = df['engagement_rate'] / np.maximum(df['hours_since_post'], 0.1)
        
        return df
    
    def _extract_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Convert timestamps if not already datetime
        if not pd.api.types.is_datetime64_any_dtype(df['vid_postTime']):
            df['vid_postTime'] = pd.to_datetime(df['vid_postTime'])
        if not pd.api.types.is_datetime64_any_dtype(df['vid_scrapeTime']):
            df['vid_scrapeTime'] = pd.to_datetime(df['vid_scrapeTime'])
        
        # Time since posting (in hours) - use existing column if available
        if 'vid_existtime_hrs' in df.columns:
            df['hours_since_post'] = df['vid_existtime_hrs']
        else:
            df['hours_since_post'] = (df['vid_scrapeTime'] - df['vid_postTime']).dt.total_seconds() / 3600
        
        df['hours_since_post'] = np.maximum(df['hours_since_post'], 0.01)

        df['total_engagement'] = df.get('vid_nlike', 0) + df.get('vid_ncomment', 0) + df.get('vid_nshare', 0) + df.get('vid_nsave', 0)
        
        # Posting time features - use existing if available
        if 'post_hour' not in df.columns:
            df['post_hour'] = df['vid_postTime'].dt.hour
        if 'post_day' not in df.columns:
            df['post_day_of_week'] = df['vid_postTime'].dt.dayofweek
        else:
            # Convert day names to numbers
            day_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 
                      'Friday': 4, 'Saturday': 5, 'Sunday': 6}
            df['post_day_of_week'] = df['post_day'].map(day_map).fillna(0)
        
        df['post_is_weekend'] = df['post_day_of_week'].isin([5, 6]).astype(int)
        
        # Optimal posting time indicators
        df['is_prime_time'] = df['post_hour'].isin([18, 19, 20, 21]).astype(int)  # 6-9 PM
        df['is_morning_peak'] = df['post_hour'].isin([7, 8, 9]).astype(int)      # 7-9 AM
        df['is_lunch_time'] = df['post_hour'].isin([12, 13]).astype(int)         # Lunch hour
        
        # Day type features
        df['is_friday'] = (df['post_day_of_week'] == 4).astype(int)
        df['is_monday'] = (df['post_day_of_week'] == 0).astype(int)
        
        # Time since posting features (different scales)
        df['log_hours_since_post'] = np.log1p(df['hours_since_post'])
        df['sqrt_hours_since_post'] = np.sqrt(df['hours_since_post'])
        
        # Growth rates (per hour) with log transformation for stability
        df['view_growth_rate'] = df['vid_nview'] / df['hours_since_post']
        df['like_growth_rate'] = df['vid_nlike'] / df['hours_since_post']
        df['comment_growth_rate'] = df['vid_ncomment'] / df['hours_since_post']
        df['share_growth_rate'] = df['vid_nshare'] / df['hours_since_post']
        df['save_growth_rate'] = df['vid_nsave'] / df['hours_since_post']
        df['engagement_growth_rate'] = df['total_engagement'] / df['hours_since_post']
        
        # Log-transformed growth rates for better distribution
        df['log_view_growth_rate'] = np.log1p(df['view_growth_rate'])
        df['log_like_growth_rate'] = np.log1p(df['like_growth_rate'])
        df['log_comment_growth_rate'] = np.log1p(df['comment_growth_rate'])
        df['log_share_growth_rate'] = np.log1p(df['share_growth_rate'])
        df['log_save_growth_rate'] = np.log1p(df['save_growth_rate'])
        df['log_engagement_growth_rate'] = np.log1p(df['engagement_growth_rate'])
        
        return df
    
    def _preprocess_raw_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Handle different column naming conventions
        column_mapping = {
            'user_name': ['user_name', 'username'],
            'vid_id': ['vid_id', 'video_id'],
            'vid_caption': ['vid_caption', 'vid_desc', 'vid_desc_clean'],
            'vid_hashtags': ['vid_hashtags', 'vid_hashtag', 'vid_hashtags_normalized'],
            'vid_duration': ['vid_duration', 'vid_duration_sec'],
            'music_nused': ['music_nused', 'music_video_count']
        }
        
        for standard_name, possible_names in column_mapping.items():
            for possible_name in possible_names:
                if possible_name in df.columns and standard_name not in df.columns:
                    df[standard_name] = df[possible_name]
                    break
        
        # Parse follower/following counts if they're strings
        if df['user_nfollower'].dtype == 'object':
            df['user_nfollower'] = df['user_nfollower'].apply(self._parse_count)
        if 'user_nfollowing' in df.columns and df['user_nfollowing'].dtype == 'object':
            df['user_nfollowing'] = df['user_nfollowing'].apply(self._parse_count)
        
        # Handle video duration - use preprocessed version if available
        if 'vid_duration_sec' in df.columns:
            df['vid_duration_seconds'] = df['vid_duration_sec']
        else:
            df['vid_duration_seconds'] = df['vid_duration'].apply(self._parse_duration)
        
        # Video duration categories and features
        df['is_short_video'] = (df['vid_duration_seconds'] <= 15).astype(int)
        df['is_medium_video'] = ((df['vid_duration_seconds'] > 15) & (df['vid_duration_seconds'] <= 60)).astype(int)
        df['is_long_video'] = (df['vid_duration_seconds'] > 60).astype(int)
        df['log_duration'] = np.log1p(df['vid_duration_seconds'])
        
        # Parse music popularity
        if 'music_nused' in df.columns:
            df['music_popularity'] = pd.to_numeric(df['music_nused'], errors='coerce').fillna(0)
        elif 'music_video_count' in df.columns:
            df['music_popularity'] = df['music_video_count'].apply(
                lambda x: self._parse_count(x.split()[0]) if pd.notna(x) and ' ' in str(x) else 0
            )
        else:
            df['music_popularity'] = 0
        
        # Music popularity features
        df['log_music_popularity'] = np.log1p(df['music_popularity'])
        df['is_trending_music'] = (df['music_popularity'] > df['music_popularity'].quantile(0.8)).astype(int)
        
        # Extract hashtags - handle preprocessed format
        if 'vid_hashtags_normalized' in df.columns:
            df['hashtag_list'] = df['vid_hashtags_normalized']
        else:
            df['hashtag_list'] = df['vid_hashtags'].apply(self._extract_hashtags)
        
        # Handle hashtag count - use existing if available
        if 'hashtag_count' in df.columns:
            df['num_hashtags'] = df['hashtag_count']
        else:
            df['num_hashtags'] = df['hashtag_list'].apply(len)
        
        # Hashtag features
        df['hashtag_density'] = df['num_hashtags'] / np.maximum(df['vid_duration_seconds'], 1)
        df['has_many_hashtags'] = (df['num_hashtags'] > 5).astype(int)
        df['has_few_hashtags'] = (df['num_hashtags'] <= 2).astype(int)
        
        # Caption features - use cleaned version if available
        caption_col = 'vid_desc_clean' if 'vid_desc_clean' in df.columns else 'vid_caption'
        df['caption_length'] = df[caption_col].fillna('').astype(str).apply(len)
        df['has_caption'] = (df['caption_length'] > 0).astype(int)
        
        # NEW: Caption features
        df['log_caption_length'] = np.log1p(df['caption_length'])
        df['caption_word_count'] = df[caption_col].fillna('').astype(str).apply(lambda x: len(x.split()))
        df['has_long_caption'] = (df['caption_length'] > 100).astype(int)
        df['has_short_caption'] = (df['caption_length'] <= 20).astype(int)
        df['caption_hashtag_ratio'] = df['caption_length'] / np.maximum(df['num_hashtags'], 1)
        
        # User verification
        df['is_verified'] = df['user_verified'].astype(int) if 'user_verified' in df.columns else 0
        
        # User features with log transformations
        df['log_user_nfollower'] = np.log1p(df['user_nfollower'])
        df['log_user_nfollowing'] = np.log1p(df.get('user_nfollowing', 0))
        df['follower_following_ratio'] = df['user_nfollower'] / np.maximum(df.get('user_nfollowing', 1), 1)
        df['is_mega_influencer'] = (df['user_nfollower'] > 1000000).astype(int)
        df['is_micro_influencer'] = ((df['user_nfollower'] >= 10000) & (df['user_nfollower'] <= 100000)).astype(int)
        df['is_nano_influencer'] = ((df['user_nfollower'] >= 1000) & (df['user_nfollower'] < 10000)).astype(int)
        
        # Calculate engagement metrics
        df = self._extract_temporal_features(df)
        df = self._calculate_engagement_metrics(df)
        
        return df
    
    def _parse_duration(self, duration_str: str) -> float:
        if pd.isna(duration_str) or duration_str == '' or duration_str == '00:00':
            return 0.0
        
        try:
            if ':' in str(duration_str):
                parts = str(duration_str).split(':')
                if len(parts) == 2:
                    minutes, seconds = map(int, parts)
                    return minutes * 60 + seconds
                elif len(parts) == 3:
                    hours, minutes, seconds = map(int, parts)
                    return hours * 3600 + minutes * 60 + seconds
            else:
                return float(duration_str)
        except (ValueError, TypeError):
            return 0.0
    
    def _calculate_snapshot_deltas(self, group: pd.DataFrame) -> pd.DataFrame:
        """Calculate deltas between consecutive snapshots for a single video"""
        group = group.sort_values('vid_scrapeTime').reset_index(drop=True)
        
        if len(group) < self.min_snapshots:
            return pd.DataFrame()
        
        features_list = []
        
        for i in range(len(group) - 1):
            if i + 1 >= len(group):
                break
                
            current = group.iloc[i]
            next_snapshot = group.iloc[i + 1]
            
            # Time delta
            time_delta = (next_snapshot['vid_scrapeTime'] - current['vid_scrapeTime']).total_seconds() / 3600
            if time_delta <= 0:
                continue
            
            # Calculate deltas
            delta_views = next_snapshot['vid_nview'] - current['vid_nview']
            delta_likes = next_snapshot['vid_nlike'] - current['vid_nlike']
            delta_comments = next_snapshot['vid_ncomment'] - current['vid_ncomment']
            delta_shares = next_snapshot['vid_nshare'] - current['vid_nshare']
            delta_saves = next_snapshot['vid_nsave'] - current['vid_nsave']
            delta_engagement = next_snapshot['total_engagement'] - current['total_engagement']
            
            # Growth rates (per hour)
            view_growth_rate = delta_views / time_delta
            like_growth_rate = delta_likes / time_delta
            comment_growth_rate = delta_comments / time_delta
            share_growth_rate = delta_shares / time_delta
            save_growth_rate = delta_saves / time_delta
            engagement_growth_rate = delta_engagement / time_delta
            
            # Acceleration (second derivative)
            if i > 0:
                prev_view_growth = (current['vid_nview'] - group.iloc[i-1]['vid_nview']) / np.maximum(
                    (current['vid_scrapeTime'] - group.iloc[i-1]['vid_scrapeTime']).total_seconds() / 3600, 0.1)
                view_acceleration = (view_growth_rate - prev_view_growth) / time_delta
                
                prev_engagement_growth = (current['total_engagement'] - group.iloc[i-1]['total_engagement']) / np.maximum(
                    (current['vid_scrapeTime'] - group.iloc[i-1]['vid_scrapeTime']).total_seconds() / 3600, 0.1)
                engagement_acceleration = (engagement_growth_rate - prev_engagement_growth) / time_delta
            else:
                view_acceleration = 0
                engagement_acceleration = 0
            
            # Relative growth (compared to current values)
            relative_view_growth = delta_views / np.maximum(current['vid_nview'], 1)
            relative_like_growth = delta_likes / np.maximum(current['vid_nlike'], 1)
            relative_comment_growth = delta_comments / np.maximum(current['vid_ncomment'], 1)
            relative_share_growth = delta_shares / np.maximum(current['vid_nshare'], 1)
            relative_save_growth = delta_saves / np.maximum(current['vid_nsave'], 1)
            
            # Engagement mix features
            engagement_mix_likes = delta_likes / np.maximum(delta_engagement, 1)
            engagement_mix_comments = delta_comments / np.maximum(delta_engagement, 1)
            engagement_mix_shares = delta_shares / np.maximum(delta_engagement, 1)
            engagement_mix_saves = delta_saves / np.maximum(delta_engagement, 1)
            
            feature_row = {
                'user_name': current['user_name'],
                'vid_id': current['vid_id'],
                'snapshot_index': i,
                'time_delta_hours': time_delta,
                
                # Current snapshot features
                'current_views': current['vid_nview'],
                'current_likes': current['vid_nlike'],
                'current_comments': current['vid_ncomment'],
                'current_shares': current['vid_nshare'],
                'current_saves': current['vid_nsave'],
                'current_engagement': current['total_engagement'],
                'current_hours_since_post': current['hours_since_post'],
                
                # Delta features
                'delta_views': delta_views,
                'delta_likes': delta_likes,
                'delta_comments': delta_comments,
                'delta_shares': delta_shares,
                'delta_saves': delta_saves,
                'delta_engagement': delta_engagement,
                
                # Growth rate features
                'view_growth_rate': view_growth_rate,
                'like_growth_rate': like_growth_rate,
                'comment_growth_rate': comment_growth_rate,
                'share_growth_rate': share_growth_rate,
                'save_growth_rate': save_growth_rate,
                'engagement_growth_rate': engagement_growth_rate,
                
                # Advanced features
                'view_acceleration': view_acceleration,
                'engagement_acceleration': engagement_acceleration,
                'relative_view_growth': relative_view_growth,
                'relative_like_growth': relative_like_growth,
                'relative_comment_growth': relative_comment_growth,
                'relative_share_growth': relative_share_growth,
                'relative_save_growth': relative_save_growth,
                'engagement_mix_likes': engagement_mix_likes,
                'engagement_mix_comments': engagement_mix_comments,
                'engagement_mix_shares': engagement_mix_shares,
                'engagement_mix_saves': engagement_mix_saves,
                
                # Log-transformed features for stability
                'log_view_growth_rate': np.log1p(max(0, view_growth_rate)),
                'log_like_growth_rate': np.log1p(max(0, like_growth_rate)),
                'log_comment_growth_rate': np.log1p(max(0, comment_growth_rate)),
                'log_share_growth_rate': np.log1p(max(0, share_growth_rate)),
                'log_save_growth_rate': np.log1p(max(0, save_growth_rate)),
                'log_engagement_growth_rate': np.log1p(max(0, engagement_growth_rate)),
                
                # Static features (from first snapshot)
                'user_nfollower': group.iloc[0]['user_nfollower'],
                'user_nfollowing': group.iloc[0].get('user_nfollowing', 0),
                'vid_duration_seconds': group.iloc[0]['vid_duration_seconds'],
                'music_popularity': group.iloc[0]['music_popularity'],
                'num_hashtags': group.iloc[0]['num_hashtags'],
                'caption_length': group.iloc[0]['caption_length'],
                'has_caption': group.iloc[0]['has_caption'],
                'is_verified': group.iloc[0]['is_verified'],
                'post_hour': group.iloc[0]['post_hour'],
                'post_day_of_week': group.iloc[0]['post_day_of_week'],
                'post_is_weekend': group.iloc[0]['post_is_weekend'],
                'vid_caption': group.iloc[0].get('vid_desc_clean', group.iloc[0].get('vid_caption', '')),
                'vid_hashtag': str(group.iloc[0]['hashtag_list']),
                'vid_category': group.iloc[0].get('vid_category', 'unknown'),
                
                # Additional static features
                'log_user_nfollower': group.iloc[0]['log_user_nfollower'],
                'follower_following_ratio': group.iloc[0]['follower_following_ratio'],
                'is_mega_influencer': group.iloc[0]['is_mega_influencer'],
                'is_micro_influencer': group.iloc[0]['is_micro_influencer'],
                'is_nano_influencer': group.iloc[0]['is_nano_influencer'],
                'is_short_video': group.iloc[0]['is_short_video'],
                'is_medium_video': group.iloc[0]['is_medium_video'],
                'is_long_video': group.iloc[0]['is_long_video'],
                'log_duration': group.iloc[0]['log_duration'],
                'is_trending_music': group.iloc[0]['is_trending_music'],
                'hashtag_density': group.iloc[0]['hashtag_density'],
                'has_many_hashtags': group.iloc[0]['has_many_hashtags'],
                'has_long_caption': group.iloc[0]['has_long_caption'],
                'is_prime_time': group.iloc[0]['is_prime_time'],
                'is_morning_peak': group.iloc[0]['is_morning_peak']
            }
            
            features_list.append(feature_row)
        
        return pd.DataFrame(features_list)
    
    def _extract_video_level_features(self, df: pd.DataFrame) -> pd.DataFrame:
        all_features = []
        
        for (user, vid_id), group in df.groupby(['user_name', 'vid_id']):
            video_features = self._calculate_snapshot_deltas(group)
            if not video_features.empty:
                all_features.append(video_features)
        
        if not all_features:
            return pd.DataFrame()
        
        return pd.concat(all_features, ignore_index=True)
    
    def _create_training_data(self, video_features: pd.DataFrame) -> pd.DataFrame:
        if video_features.empty:
            return pd.DataFrame()
        
        training_data = []
        
        for (user, vid_id), group in video_features.groupby(['user_name', 'vid_id']):
            group = group.sort_values('snapshot_index').reset_index(drop=True)
            
            if len(group) < 3:  # Need at least 3 snapshots to predict 4th
                continue
            
            feature_snapshots = group.iloc[:2]
            
            if len(group) >= 3:
                target_snapshot = group.iloc[2]
                
                # NEW: More sophisticated aggregation features
                # Rolling statistics with different windows
                if len(group) > 2:
                    # Use all available snapshots for rolling features
                    rolling_data = group.iloc[:3]  # Use up to 3 snapshots
                    
                    # Rolling window features
                    rolling_view_growth = rolling_data['view_growth_rate'].rolling(window=2, min_periods=1)
                    rolling_engagement_growth = rolling_data['engagement_growth_rate'].rolling(window=2, min_periods=1)
                    
                    view_growth_trend = rolling_view_growth.mean().iloc[-1]
                    view_growth_volatility = rolling_view_growth.std().iloc[-1] if len(rolling_data) > 1 else 0
                    engagement_growth_trend = rolling_engagement_growth.mean().iloc[-1]
                    engagement_growth_volatility = rolling_engagement_growth.std().iloc[-1] if len(rolling_data) > 1 else 0
                else:
                    view_growth_trend = feature_snapshots['view_growth_rate'].mean()
                    view_growth_volatility = feature_snapshots['view_growth_rate'].std()
                    engagement_growth_trend = feature_snapshots['engagement_growth_rate'].mean()
                    engagement_growth_volatility = feature_snapshots['engagement_growth_rate'].std()
                
                # Momentum features (recent vs earlier)
                if len(feature_snapshots) >= 2:
                    recent_view_momentum = feature_snapshots.iloc[1]['view_growth_rate'] - feature_snapshots.iloc[0]['view_growth_rate']
                    recent_engagement_momentum = feature_snapshots.iloc[1]['engagement_growth_rate'] - feature_snapshots.iloc[0]['engagement_growth_rate']
                else:
                    recent_view_momentum = 0
                    recent_engagement_momentum = 0
                
                # Interaction features
                follower_engagement_interaction = feature_snapshots.iloc[0]['user_nfollower'] * feature_snapshots['engagement_growth_rate'].mean()
                duration_engagement_interaction = feature_snapshots.iloc[0]['vid_duration_seconds'] * feature_snapshots['engagement_growth_rate'].mean()
                hashtag_engagement_interaction = feature_snapshots.iloc[0]['num_hashtags'] * feature_snapshots['engagement_growth_rate'].mean()
                
                # Normalized features (relative to user's follower count)
                normalized_views = feature_snapshots.iloc[-1]['current_views'] / np.maximum(feature_snapshots.iloc[0]['user_nfollower'], 1)
                normalized_engagement = feature_snapshots.iloc[-1]['current_engagement'] / np.maximum(feature_snapshots.iloc[0]['user_nfollower'], 1)
                
                # Aggregate features from first 2 snapshots
                feature_row = {
                    'user_name': user,
                    'vid_id': vid_id,
                    
                    # User features
                    'user_nfollower': feature_snapshots.iloc[0]['user_nfollower'],
                    'user_nfollowing': feature_snapshots.iloc[0]['user_nfollowing'],
                    'is_verified': feature_snapshots.iloc[0]['is_verified'],
                    'log_user_nfollower': feature_snapshots.iloc[0]['log_user_nfollower'],
                    'follower_following_ratio': feature_snapshots.iloc[0]['follower_following_ratio'],
                    'is_mega_influencer': feature_snapshots.iloc[0]['is_mega_influencer'],
                    'is_micro_influencer': feature_snapshots.iloc[0]['is_micro_influencer'],
                    'is_nano_influencer': feature_snapshots.iloc[0]['is_nano_influencer'],
                    
                    # Video content features
                    'vid_duration_seconds': feature_snapshots.iloc[0]['vid_duration_seconds'],
                    'log_duration': feature_snapshots.iloc[0]['log_duration'],
                    'is_short_video': feature_snapshots.iloc[0]['is_short_video'],
                    'is_medium_video': feature_snapshots.iloc[0]['is_medium_video'],
                    'is_long_video': feature_snapshots.iloc[0]['is_long_video'],
                    'music_popularity': feature_snapshots.iloc[0]['music_popularity'],
                    'is_trending_music': feature_snapshots.iloc[0]['is_trending_music'],
                    'num_hashtags': feature_snapshots.iloc[0]['num_hashtags'],
                    'hashtag_density': feature_snapshots.iloc[0]['hashtag_density'],
                    'has_many_hashtags': feature_snapshots.iloc[0]['has_many_hashtags'],
                    'caption_length': feature_snapshots.iloc[0]['caption_length'],
                    'has_caption': feature_snapshots.iloc[0]['has_caption'],
                    'has_long_caption': feature_snapshots.iloc[0]['has_long_caption'],
                    'post_hour': feature_snapshots.iloc[0]['post_hour'],
                    'post_day_of_week': feature_snapshots.iloc[0]['post_day_of_week'],
                    'post_is_weekend': feature_snapshots.iloc[0]['post_is_weekend'],
                    'is_prime_time': feature_snapshots.iloc[0]['is_prime_time'],
                    'is_morning_peak': feature_snapshots.iloc[0]['is_morning_peak'],
                    
                    # Text features
                    'vid_caption': feature_snapshots.iloc[0]['vid_caption'],
                    'vid_hashtag': feature_snapshots.iloc[0]['vid_hashtag'],
                    'vid_category': feature_snapshots.iloc[0]['vid_category'],
                    
                    # Time series features 
                    'avg_view_growth_rate': feature_snapshots['view_growth_rate'].mean(),
                    'avg_like_growth_rate': feature_snapshots['like_growth_rate'].mean(),
                    'avg_comment_growth_rate': feature_snapshots['comment_growth_rate'].mean(),
                    'avg_share_growth_rate': feature_snapshots['share_growth_rate'].mean(),
                    'avg_save_growth_rate': feature_snapshots['save_growth_rate'].mean(),
                    'avg_engagement_growth_rate': feature_snapshots['engagement_growth_rate'].mean(),
                    
                    'std_view_growth_rate': feature_snapshots['view_growth_rate'].std(),
                    'std_like_growth_rate': feature_snapshots['like_growth_rate'].std(),
                    'std_comment_growth_rate': feature_snapshots['comment_growth_rate'].std(),
                    'std_share_growth_rate': feature_snapshots['share_growth_rate'].std(),
                    'std_save_growth_rate': feature_snapshots['save_growth_rate'].std(),
                    'std_engagement_growth_rate': feature_snapshots['engagement_growth_rate'].std(),

                    'max_view_growth_rate': feature_snapshots['view_growth_rate'].max(),
                    'min_view_growth_rate': feature_snapshots['view_growth_rate'].min(),
                    'max_engagement_growth_rate': feature_snapshots['engagement_growth_rate'].max(),
                    'min_engagement_growth_rate': feature_snapshots['engagement_growth_rate'].min(),
                    
                    # Trend and momentum features
                    'view_growth_trend': view_growth_trend,
                    'view_growth_volatility': view_growth_volatility,
                    'engagement_growth_trend': engagement_growth_trend,
                    'engagement_growth_volatility': engagement_growth_volatility,
                    'recent_view_momentum': recent_view_momentum,
                    'recent_engagement_momentum': recent_engagement_momentum,
                    
                    # Interaction features
                    'follower_engagement_interaction': follower_engagement_interaction,
                    'duration_engagement_interaction': duration_engagement_interaction,
                    'hashtag_engagement_interaction': hashtag_engagement_interaction,
                    
                    # Normalized features
                    'normalized_views': normalized_views,
                    'normalized_engagement': normalized_engagement,
                    
                    # Acceleration features
                    'avg_view_acceleration': feature_snapshots['view_acceleration'].mean(),
                    'avg_engagement_acceleration': feature_snapshots['engagement_acceleration'].mean(),
                    
                    # Relative growth features
                    'avg_relative_view_growth': feature_snapshots['relative_view_growth'].mean(),
                    'avg_relative_like_growth': feature_snapshots['relative_like_growth'].mean(),
                    'avg_relative_comment_growth': feature_snapshots['relative_comment_growth'].mean(),
                    'avg_relative_share_growth': feature_snapshots['relative_share_growth'].mean(),
                    'avg_relative_save_growth': feature_snapshots['relative_save_growth'].mean(),
                    
                    # Engagement mix features
                    'avg_engagement_mix_likes': feature_snapshots['engagement_mix_likes'].mean(),
                    'avg_engagement_mix_comments': feature_snapshots['engagement_mix_comments'].mean(),
                    'avg_engagement_mix_shares': feature_snapshots['engagement_mix_shares'].mean(),
                    'avg_engagement_mix_saves': feature_snapshots['engagement_mix_saves'].mean(),
                    
                    # Latest snapshot values
                    'latest_views': feature_snapshots.iloc[-1]['current_views'],
                    'latest_likes': feature_snapshots.iloc[-1]['current_likes'],
                    'latest_comments': feature_snapshots.iloc[-1]['current_comments'],
                    'latest_shares': feature_snapshots.iloc[-1]['current_shares'],
                    'latest_saves': feature_snapshots.iloc[-1]['current_saves'],
                    'latest_hours_since_post': feature_snapshots.iloc[-1]['current_hours_since_post'],
                    
                    # Target variables (growth rates to predict) 
                    'target_view_growth_rate': self._stabilize_target(target_snapshot['view_growth_rate']),
                    'target_like_growth_rate': self._stabilize_target(target_snapshot['like_growth_rate']),
                    'target_comment_growth_rate': self._stabilize_target(target_snapshot['comment_growth_rate']),
                    'target_share_growth_rate': self._stabilize_target(target_snapshot['share_growth_rate']),
                    'target_save_growth_rate': self._stabilize_target(target_snapshot['save_growth_rate']),
                    'target_engagement_growth_rate': self._stabilize_target(target_snapshot['engagement_growth_rate']),
                    
                    # Log-transformed targets for better distribution
                    'target_log_view_growth_rate': np.log1p(max(0, target_snapshot['view_growth_rate'])),
                    'target_log_like_growth_rate': np.log1p(max(0, target_snapshot['like_growth_rate'])),
                    'target_log_comment_growth_rate': np.log1p(max(0, target_snapshot['comment_growth_rate'])),
                    'target_log_share_growth_rate': np.log1p(max(0, target_snapshot['share_growth_rate'])),
                    'target_log_save_growth_rate': np.log1p(max(0, target_snapshot['save_growth_rate'])),
                    'target_log_engagement_growth_rate': np.log1p(max(0, target_snapshot['engagement_growth_rate'])),
                    
                    # Growth classification targets
                    'view_growth_class': self._classify_growth(target_snapshot['view_growth_rate']),
                    'like_growth_class': self._classify_growth(target_snapshot['like_growth_rate']),
                    'comment_growth_class': self._classify_growth(target_snapshot['comment_growth_rate']),
                    'share_growth_class': self._classify_growth(target_snapshot['share_growth_rate']),
                    'save_growth_class': self._classify_growth(target_snapshot['save_growth_rate']),
                    'engagement_growth_class': self._classify_growth(target_snapshot['engagement_growth_rate'])
                }
                
                training_data.append(feature_row)
        
        return pd.DataFrame(training_data)
    
    def _stabilize_target(self, value: float, cap_percentile: float = 95) -> float:
        """Stabilize target values by capping extreme outliers"""
        if pd.isna(value):
            return 0.0
        
        # Cap extreme values to reduce impact of outliers
        if hasattr(self, '_target_caps'):
            cap_value = self._target_caps.get('cap', np.inf)
            return np.clip(value, -cap_value, cap_value)
        
        return value
    
    def _classify_growth(self, growth_rate: float, 
                        increase_threshold: float = 5.0,  # Reduced threshold for more balanced classes
                        decrease_threshold: float = -2.0) -> str:
        """Classify growth rate into categories: increase/stable/decrease"""
        if pd.isna(growth_rate):
            return 'stable'
        
        if growth_rate > increase_threshold:
            return 'increase'
        elif growth_rate < decrease_threshold:
            return 'decrease'
        else:
            return 'stable'
    
    def _extract_user_trend_features(self, training_data: pd.DataFrame) -> pd.DataFrame:
        if training_data.empty:
            return pd.DataFrame()

        user_features = []

        for user, group in training_data.groupby('user_name'):
            group = group.sort_values('latest_hours_since_post', ascending=False).head(self.n_recent)
            if len(group) == 0:
                continue

            # Precompute stats to avoid repetition
            caption_mean = group['caption_length'].mean()
            duration_mean = group['vid_duration_seconds'].mean()
            engagement_quality = group['engagement_quality'] if 'engagement_quality' in group.columns else group['avg_engagement_growth_rate']
            engagement_velocity = group['engagement_velocity'] if 'engagement_velocity' in group.columns else group['avg_engagement_growth_rate'] / group['latest_hours_since_post']

            user_stats = {
                'user_name': user,
                'num_videos': len(group),

                # User profile features
                'avg_follower_count': group['user_nfollower'].mean(),
                'avg_following_count': group['user_nfollowing'].mean(),
                'verification_rate': group['is_verified'].mean(),
                'avg_follower_following_ratio': group['follower_following_ratio'].mean(),
                'influencer_tier_score': (
                    group['is_mega_influencer'].mean() * 4 +
                    group['is_micro_influencer'].mean() * 2 +
                    group['is_nano_influencer'].mean() * 1
                ),

                # Content features
                'avg_video_duration': duration_mean,
                'std_video_duration': group['vid_duration_seconds'].std(),
                'short_video_rate': group['is_short_video'].mean(),
                'long_video_rate': group['is_long_video'].mean(),
                'avg_hashtag_count': group['num_hashtags'].mean(),
                'avg_caption_length': caption_mean,
                'caption_usage_rate': group['has_caption'].mean(),
                'long_caption_rate': group['has_long_caption'].mean(),
                'trending_music_rate': group['is_trending_music'].mean(),

                # Performance trends
                'avg_view_growth': group['avg_view_growth_rate'].mean(),
                'avg_like_growth': group['avg_like_growth_rate'].mean(),
                'avg_comment_growth': group['avg_comment_growth_rate'].mean(),
                'avg_share_growth': group['avg_share_growth_rate'].mean(),
                'avg_save_growth': group['avg_save_growth_rate'].mean(),
                'avg_engagement_growth': group['avg_engagement_growth_rate'].mean(),

                # Performance variability
                'std_view_growth': group['avg_view_growth_rate'].std(),
                'std_like_growth': group['avg_like_growth_rate'].std(),
                'std_engagement_growth': group['avg_engagement_growth_rate'].std(),

                # Consistency metrics
                'view_growth_consistency': 1 / (1 + group['std_view_growth_rate'].mean()),
                'like_growth_consistency': 1 / (1 + group['std_like_growth_rate'].mean()),
                'engagement_growth_consistency': 1 / (1 + group['std_engagement_growth_rate'].mean()),

                # Performance percentiles
                'view_growth_75th': group['avg_view_growth_rate'].quantile(0.75),
                'view_growth_25th': group['avg_view_growth_rate'].quantile(0.25),
                'engagement_growth_75th': group['avg_engagement_growth_rate'].quantile(0.75),
                'engagement_growth_25th': group['avg_engagement_growth_rate'].quantile(0.25),

                # Success rate
                'view_success_rate': (group['target_view_growth_rate'] > 0).mean(),
                'like_success_rate': (group['target_like_growth_rate'] > 0).mean(),
                'engagement_success_rate': (group['target_engagement_growth_rate'] > 0).mean(),

                # High performance
                'high_view_performance_rate': (
                    (group['target_view_growth_rate'] > group['target_view_growth_rate'].median()).mean()
                ),
                'high_engagement_performance_rate': (
                    (group['target_engagement_growth_rate'] > group['target_engagement_growth_rate'].median()).mean()
                ),

                # Growth class distribution
                'increase_rate': (group['engagement_growth_class'] == 'increase').mean(),
                'stable_rate': (group['engagement_growth_class'] == 'stable').mean(),
                'decrease_rate': (group['engagement_growth_class'] == 'decrease').mean(),

                # Posting patterns
                'avg_post_hour': group['post_hour'].mean(),
                'weekend_posting_rate': group['post_is_weekend'].mean(),
                'prime_time_posting_rate': group['is_prime_time'].mean(),
                'morning_peak_posting_rate': group['is_morning_peak'].mean(),
                'posting_hour_diversity': group['post_hour'].nunique(),

                # Recent performance
                'recent_avg_views': group['latest_views'].mean(),
                'recent_avg_engagement': (
                    group['latest_likes'] + group['latest_comments'] +
                    group['latest_shares'] + group['latest_saves']
                ).mean(),
                'recent_max_views': group['latest_views'].max(),
                'recent_max_engagement': (
                    group['latest_likes'] + group['latest_comments'] +
                    group['latest_shares'] + group['latest_saves']
                ).max(),

                # Advanced behavior
                'normalized_avg_views': group['normalized_views'].mean(),
                'normalized_avg_engagement': group['normalized_engagement'].mean(),
                'avg_momentum_view': group['recent_view_momentum'].mean(),
                'avg_momentum_engagement': group['recent_engagement_momentum'].mean(),

                # Content strategy
                'hashtag_strategy_score': group['hashtag_density'].mean(),
                'content_length_strategy': duration_mean / caption_mean if caption_mean > 0 else 0,

                # Engagement quality
                'avg_engagement_quality': engagement_quality.mean(),
                'engagement_velocity': engagement_velocity.mean(),
            }

            user_features.append(user_stats)

        return pd.DataFrame(user_features)

    def transform(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        processed_df = self._preprocess_raw_data(df)
        video_features = self._extract_video_level_features(processed_df)
        
        if video_features.empty:
            print("Warning: No video features extracted!")
            return {
                'training_data': pd.DataFrame(),
                'user_trend_features': pd.DataFrame(),
                'video_features': pd.DataFrame(),
                'text_features': np.array([]),
                'text_vectorizer': None
            }
        
        training_data = self._create_training_data(video_features)
        
        if not training_data.empty:
            # Calculate target caps for stabilization
            target_cols = [col for col in training_data.columns if col.startswith('target_') and 'log' not in col and 'class' not in col]
            self._target_caps = {}
            for col in target_cols:
                cap_value = training_data[col].quantile(0.95)
                self._target_caps[col] = cap_value
            self._target_caps['cap'] = max(self._target_caps.values()) if self._target_caps else 1000
            
            # Apply stabilization to existing targets
            for col in target_cols:
                if col in training_data.columns:
                    training_data[col] = training_data[col].apply(self._stabilize_target)
        
        user_trend_features = self._extract_user_trend_features(training_data)
        text_features = self._prepare_text_features(training_data)

        return {
            'training_data': training_data,
            'user_trend_features': user_trend_features,
            'video_features': video_features,
            'text_features': text_features,
            'text_vectorizer': self.text_vectorizer
        }