# feature_engineering.py

import pandas as pd
import numpy as np
from typing import Dict


class TikTokFeatureEngineer:
    def __init__(self, n_recent_videos: int = 20):
        self.n_recent = n_recent_videos

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:

        # Số giờ kể từ thời điểm đăng đến lúc được crawl 
        df['vid_existtime_hrs'] = df['vid_existtime_hrs'].clip(lower=1e-3)

        # tổng số tương tác (like + comment + share + save).
        df['vid_engagement'] = df[['vid_nlike', 'vid_ncomment', 'vid_nshare', 'vid_nsave']].sum(axis=1)
        # Tỷ lệ tương tác / view 
        df['vid_engagement_rate'] = df['vid_engagement'] / df['vid_nview'].replace(0, np.nan)
        # Tốc độ tăng view theo từng giờ 
        df['vid_view_growth_rate'] = df['vid_nview'] / df['vid_existtime_hrs']

        return df

    # Tính tốc độ tăng trưởng tương tác cho từng video, dựa trên các snapshot được crawl theo thời gian (4 dòng liền kề trong input) 
    def _extract_video_features(self, df: pd.DataFrame) -> pd.DataFrame:
        all_features = []

        for (user, vid), group in df.groupby(['user_name', 'vid_id']):
            #  Gom 4 snapshot của các vid_id lại 
            group = group.sort_values('vid_scrapeTime')
            if len(group) < 4:
                continue

            # 3 snapshot đầu (t0, t1, t2)
            X_group = group.iloc[:3]
            cols_to_diff = ['vid_nview', 'vid_nlike', 'vid_ncomment', 'vid_nshare', 'vid_nsave']
            delta = X_group[cols_to_diff].diff().fillna(0)
            time_diff = X_group['vid_existtime_hrs'].diff().fillna(0)
            time_diff = time_diff.replace(0, np.nan)
 

            features = {
                'user_name': user,
                'vid_id': vid,
                'vid_postTime' : group.iloc[-1]['vid_postTime'],
                'vid_duration': group.iloc[-1]['vid_duration'],
                'vid_view_growth_per_hour': (delta['vid_nview'] / time_diff).mean(),
                'vid_like_growth_per_hour': (delta['vid_nlike'] / time_diff).mean(),
                'vid_comment_growth_per_hour': (delta['vid_ncomment'] / time_diff).mean(),
                'vid_share_growth_per_hour': (delta['vid_nshare'] / time_diff).mean(),
                'vid_save_growth_per_hour': (delta['vid_nsave'] / time_diff).mean(),
                'vid_engagement_growth_per_hour': (delta[['vid_nlike','vid_ncomment','vid_nshare','vid_nsave']].sum(axis=1) / time_diff).mean(),
                #vid_engagement_rate_change': (delta[df['vid_engagement'] / df['vid_nview']]).mean()
            }


            # Snapshot thứ 4 (t3) – dùng làm nhãn
            t2 = group.iloc[2]
            t3 = group.iloc[3]
            delta_views = t3['vid_nview'] - t2['vid_nview']
            delta_time = (t3['vid_existtime_hrs'] - t2['vid_existtime_hrs'])
            if delta_time == 0:
                continue

            target_growth = delta_views / delta_time
            features['future_view_growth_per_hour'] = target_growth

             # Tính engagement rate change
            delta_engagement_rate = X_group['vid_engagement_rate'].diff().fillna(0)
            features['vid_engagement_rate_change'] = delta_engagement_rate.mean()

            all_features.append(features)

        return pd.DataFrame(all_features)

    def _extract_user_trend_features(self, video_features: pd.DataFrame) -> pd.DataFrame:
        user_features = []

        for user, group in video_features.groupby('user_name'):
            group = group.sort_values('vid_postTime', ascending=False).head(self.n_recent)

            agg = {
                'user_name': user,
                'user_avg_view_growth': group['vid_view_growth_per_hour'].mean(),
                'user_avg_like_growth': group['vid_like_growth_per_hour'].mean(),
                'user_avg_comment_growth': group['vid_comment_growth_per_hour'].mean(),
                'user_avg_share_growth': group['vid_share_growth_per_hour'].mean(),
                'user_avg_engagement_growth': group['vid_engagement_growth_per_hour'].mean(),
                'user_avg_engagement_rate_change': group['vid_engagement_rate_change'].mean()
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

def main():
    infile = r"D:\UIT\DS200\DS2000 Project\Dataset\Preprocessed Data\training_data.csv"
    feature_outfile = r"D:\UIT\DS200\DS2000 Project\Dataset\FE Results\Video_Feature.csv"
    trend_outfile = r"D:\UIT\DS200\DS2000 Project\Dataset\FE Results\Trend_Feature.csv"

    # Đọc dữ liệu gốc từ file CSV
    df = pd.read_csv(infile)

    # Tạo đối tượng feature engineer
    engineer = TikTokFeatureEngineer(n_recent_videos=20)

    # Trích xuất đặc trưng
    features = engineer.transform(df)

    features['video_features'].to_csv(feature_outfile, index=False)
    features['user_trend_features'].to_csv(trend_outfile, index=False)


if __name__ == "__main__":
    main()