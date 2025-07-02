import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import re
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

class TikTokFeatureEngineerOnline:
    def __init__(self, n_recent_videos: int = 20, min_snapshots: int = 3, max_text_features: int = 512):
        self.n_recent = n_recent_videos
        self.min_snapshots = min_snapshots
        self.max_text_features = max_text_features
        self.feature_columns = []
        self.text_vectorizer = None
        
    def _parse_count(self, count_str: str) -> float:
        """Parse count strings like '14.7K', '3.9M', '72K'"""
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
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95
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
        
        df['hours_since_post'] = np.maximum(df['hours_since_post'], 0.01)  # Avoid zero
        
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
        
        # Scraping time features
        df['scrape_hour'] = df['vid_scrapeTime'].dt.hour
        df['scrape_day_of_week'] = df['vid_scrapeTime'].dt.dayofweek
        
        # Growth rates (per hour)
        df['view_growth_rate'] = df['vid_nview'] / df['hours_since_post']
        df['like_growth_rate'] = df['vid_nlike'] / df['hours_since_post']
        df['comment_growth_rate'] = df['vid_ncomment'] / df['hours_since_post']
        df['share_growth_rate'] = df['vid_nshare'] / df['hours_since_post']
        df['save_growth_rate'] = df['vid_nsave'] / df['hours_since_post']
        df['engagement_growth_rate'] = df['total_engagement'] / df['hours_since_post']
        
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
        
        # Parse music popularity
        if 'music_nused' in df.columns:
            df['music_popularity'] = pd.to_numeric(df['music_nused'], errors='coerce').fillna(0)
        elif 'music_video_count' in df.columns:
            df['music_popularity'] = df['music_video_count'].apply(
                lambda x: self._parse_count(x.split()[0]) if pd.notna(x) and ' ' in str(x) else 0
            )
        else:
            df['music_popularity'] = 0
        
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
        
        # Caption features - use cleaned version if available
        caption_col = 'vid_desc_clean' if 'vid_desc_clean' in df.columns else 'vid_caption'
        df['caption_length'] = df[caption_col].fillna('').astype(str).apply(len)
        df['has_caption'] = (df['caption_length'] > 0).astype(int)
        
        # User verification
        df['is_verified'] = df['user_verified'].astype(int) if 'user_verified' in df.columns else 0
        
        # Calculate engagement metrics
        df = self._calculate_engagement_metrics(df)
        df = self._extract_temporal_features(df)
        
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
                'vid_category': group.iloc[0].get('vid_category', 'unknown')
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
    
    def _create_inference_data(self, video_features: pd.DataFrame) -> pd.DataFrame:
        if video_features.empty:
            return pd.DataFrame()
        
        inference_data = []
        
        for (user, vid_id), group in video_features.groupby(['user_name', 'vid_id']):
            group = group.sort_values('snapshot_index').reset_index(drop=True)
            
            if len(group) < 3:  # Need at least 3 snapshots to predict 4th
                continue
            
            feature_snapshots = group.iloc[:3]
            latest_snapshot = group.iloc[2]
            
            
                
           
            feature_row = {
                'user_name': user,
                'vid_id': vid_id,
                    
                # User features
                'user_nfollower': feature_snapshots.iloc[0]['user_nfollower'],
                'user_nfollowing': feature_snapshots.iloc[0]['user_nfollowing'],
                'is_verified': feature_snapshots.iloc[0]['is_verified'],
                    
                # Video content features
                'vid_duration_seconds': feature_snapshots.iloc[0]['vid_duration_seconds'],
                'music_popularity': feature_snapshots.iloc[0]['music_popularity'],
                'num_hashtags': feature_snapshots.iloc[0]['num_hashtags'],
                'caption_length': feature_snapshots.iloc[0]['caption_length'],
                'has_caption': feature_snapshots.iloc[0]['has_caption'],
                'post_hour': feature_snapshots.iloc[0]['post_hour'],
                'post_day_of_week': feature_snapshots.iloc[0]['post_day_of_week'],
                'post_is_weekend': feature_snapshots.iloc[0]['post_is_weekend'],
                    
                # Text features
                'vid_caption': feature_snapshots.iloc[0]['vid_caption'],
                'vid_hashtag': feature_snapshots.iloc[0]['vid_hashtag'],
                'vid_category': feature_snapshots.iloc[0]['vid_category'],
                    
                # Time series features (aggregated from first 2 snapshots)
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
                    
                # Latest snapshot values
                'latest_views': latest_snapshot['current_views'],
                'latest_likes': latest_snapshot['current_likes'],
                'latest_comments': latest_snapshot['current_comments'],
                'latest_shares': latest_snapshot['current_shares'],
                'latest_saves': latest_snapshot['current_saves'],
                'latest_hours_since_post': latest_snapshot['current_hours_since_post'],
                    
            }
                
            inference_data.append(feature_row)
        
        return pd.DataFrame(inference_data)
    
    def _classify_growth(self, growth_rate: float, 
                        increase_threshold: float = 10.0, 
                        decrease_threshold: float = -5.0) -> str:
        """Classify growth rate into categories: increase/stable/decrease"""
        if pd.isna(growth_rate):
            return 'stable'
        
        if growth_rate > increase_threshold:
            return 'increase'
        elif growth_rate < decrease_threshold:
            return 'decrease'
        else:
            return 'stable'
    
    def _extract_user_trend_features(self, inference_data: pd.DataFrame) -> pd.DataFrame:
        if inference_data.empty:
            return pd.DataFrame()
        
        user_features = []
        
        for user, group in inference_data.groupby('user_name'):
            # Sort by latest activity
            group = group.sort_values('latest_hours_since_post', ascending=False).head(self.n_recent)
            
            if len(group) == 0:
                continue
            
            user_stats = {
                'user_name': user,
                'num_videos': len(group),
                
                # User profile features
                'avg_follower_count': group['user_nfollower'].mean(),
                'avg_following_count': group['user_nfollowing'].mean(),
                'verification_rate': group['is_verified'].mean(),
                
                # Content features
                'avg_video_duration': group['vid_duration_seconds'].mean(),
                'avg_hashtag_count': group['num_hashtags'].mean(),
                'avg_caption_length': group['caption_length'].mean(),
                'caption_usage_rate': group['has_caption'].mean(),
                
                # Performance trends
                'avg_view_growth': group['avg_view_growth_rate'].mean(),
                'avg_like_growth': group['avg_like_growth_rate'].mean(),
                'avg_comment_growth': group['avg_comment_growth_rate'].mean(),
                'avg_share_growth': group['avg_share_growth_rate'].mean(),
                'avg_save_growth': group['avg_save_growth_rate'].mean(),
                'avg_engagement_growth': group['avg_engagement_growth_rate'].mean(),
                
                # Consistency metrics
                'view_growth_consistency': 1 / (1 + group['std_view_growth_rate'].mean()),
                'like_growth_consistency': 1 / (1 + group['std_like_growth_rate'].mean()),
                'engagement_growth_consistency': 1 / (1 + group['std_engagement_growth_rate'].mean()),
                
                
                # Posting patterns
                'avg_post_hour': group['post_hour'].mean(),
                'weekend_posting_rate': group['post_is_weekend'].mean(),
                
                # Recent performance
                'recent_avg_views': group['latest_views'].mean(),
                'recent_avg_engagement': (group['latest_likes'] + group['latest_comments'] + 
                                        group['latest_shares'] + group['latest_saves']).mean()
            }
            
            user_features.append(user_stats)
        
        return pd.DataFrame(user_features)
    
    def transform(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        processed_df = self._preprocess_raw_data(df)
        video_features = self._extract_video_level_features(processed_df)
        
        if video_features.empty:
            print("Warning: No video features extracted!")
            return {
                'inference_data': pd.DataFrame(),
                'user_trend_features': pd.DataFrame(),
                'video_features': pd.DataFrame(),
                'text_features': np.array([]),
                'text_vectorizer': None
            }
        
        inference_data = self._create_inference_data(video_features)
        user_trend_features = self._extract_user_trend_features(inference_data)
        text_features = self._prepare_text_features(inference_data)

        return {
            'inference_data': inference_data,
            'user_trend_features': user_trend_features,
            'video_features': video_features,
            'text_features': text_features,
            'text_vectorizer': self.text_vectorizer
        }
