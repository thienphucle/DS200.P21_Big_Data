import pandas as pd
import numpy as np
from typing import Dict


class TikTokFeatureEngineerOnline:
    def __init__(self, n_recent_videos: int = 20):
        self.n_recent = n_recent_videos

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df['vid_postTime'] = pd.to_datetime(df['vid_postTime'])
        df['vid_scrapeTime'] = pd.to_datetime(df['vid_scrapeTime'])

        # Số giờ kể từ thời điểm đăng đến lúc được crawl
        df['hour_since_post'] = (df['vid_scrapeTime'] - df['vid_postTime']).dt.total_seconds() / 3600.0
        df['hour_since_post'] = df['hour_since_post'].clip(lower=1e-3)

        # Tổng số tương tác (like + comment + share + save)
        df['engagement'] = df[['vid_nlike', 'vid_ncomment', 'vid_nshare', 'vid_nsave']].sum(axis=1)

        # Tỷ lệ tương tác / lượt xem
        df['engagement_rate'] = df['engagement'] / df['vid_nview'].replace(0, np.nan)

        # Tốc độ tăng view theo từng giờ
        df['growth_rate'] = df['vid_nview'] / df['hour_since_post']

        return df

    def _extract_video_features(self, df: pd.DataFrame) -> pd.DataFrame:
        all_features = []

        for (user, vid), group in df.groupby(['user_name', 'vid_id']):
            group = group.sort_values('vid_scrapeTime')
            if len(group) < 3:
                continue

            X_group = group.iloc[:3]
            delta = X_group.diff().fillna(0)
            time_diff = X_group['vid_scrapeTime'].diff().dt.total_seconds().fillna(0) / 3600.0
            time_diff = time_diff.replace(0, np.nan)

            features = {
                'user_name': user,
                'vid_id': vid,
                'vid_duration': group.iloc[-1]['vid_duration'],
                'vid_postTime': group.iloc[-1]['vid_postTime'],  # cần để trích xuất xu hướng người dùng
                'view_growth_per_hour': (delta['vid_nview'] / time_diff).mean(),
                'like_growth_per_hour': (delta['vid_nlike'] / time_diff).mean(),
                'comment_growth_per_hour': (delta['vid_ncomment'] / time_diff).mean(),
                'share_growth_per_hour': (delta['vid_nshare'] / time_diff).mean(),
                'save_growth_per_hour': (delta['vid_nsave'] / time_diff).mean(),
                'engagement_growth_per_hour': (
                    delta[['vid_nlike', 'vid_ncomment', 'vid_nshare', 'vid_nsave']].sum(axis=1) / time_diff
                ).mean(),
            }

            all_features.append(features)

        return pd.DataFrame(all_features)

    def _extract_user_trend_features(self, video_features: pd.DataFrame) -> pd.DataFrame:
        user_features = []

        for user, group in video_features.groupby('user_name'):
            group = group.sort_values('vid_postTime', ascending=False).head(self.n_recent)

            agg = {
                'user_name': user,
                'avg_view_growth': group['view_growth_per_hour'].mean(),
                'avg_like_growth': group['like_growth_per_hour'].mean(),
                'avg_comment_growth': group['comment_growth_per_hour'].mean(),
                'avg_share_growth': group['share_growth_per_hour'].mean(),
                'avg_engagement_growth': group['engagement_growth_per_hour'].mean(),
            }
            user_features.append(agg)

        return pd.DataFrame(user_features)

    def transform(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        df = self._preprocess(df)
        video_features = self._extract_video_features(df)
        user_trend_features = self._extract_user_trend_features(video_features)

        return {
            'video_features': video_features,
            'user_trend_features': user_trend_features
        }