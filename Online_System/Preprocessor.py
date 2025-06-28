import pandas as pd
import numpy as np
from datetime import datetime
import emoji
import re
from dateutil import tz
from pyvi import ViTokenizer

class TikTokPreprocessor:
    def __init__(self, stopword_path: r"D:\UIT\DS200\DS2000 Project\Offline System\vietnamese-stopwords-dash.txt"):
        self.stopwords = set()
        if stopword_path:
            self.load_stopwords(stopword_path)

    def load_stopwords(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            self.stopwords = set(line.strip() for line in f)

    def remove_emoji(self, text):
        return emoji.replace_emoji(text, '')

    def remove_stopwords(self, text):
        words = ViTokenizer.tokenize(text).split()
        return ' '.join([w for w in words if w.lower() not in self.stopwords])

    def normalize_hashtags(self, hashtags):
        if pd.isna(hashtags) or hashtags == '':
            return []
        return [tag.strip().lower() for tag in hashtags.split(',')]

    def convert_to_vietnam_time(self, utc_time):
        from_zone = tz.tzutc()
        to_zone = tz.gettz('Asia/Ho_Chi_Minh')
        utc = pd.to_datetime(utc_time).replace(tzinfo=from_zone)
        return utc.astimezone(to_zone)

    def convert_units(self, x):
        if isinstance(x, str):
            x = x.strip().upper().replace(' VIDEOS', '')
            try:
                if 'K' in x:
                    return float(x.replace('K', '').replace(',', '')) * 1_000
                elif 'M' in x:
                    return float(x.replace('M', '').replace(',', '')) * 1_000_000
                elif 'B' in x:
                    return float(x.replace('B', '').replace(',', '')) * 1_000_000_000
                else:
                    return float(x.replace(',', ''))
            except ValueError:
                return np.nan
        elif isinstance(x, (int, float)):
            return x
        return np.nan

    def duration_to_seconds(self, duration):
        if isinstance(duration, str) and ':' in duration:
            parts = duration.split(':')
            if len(parts) == 2:
                return int(parts[0]) * 60 + int(parts[1])
            elif len(parts) == 3:
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        return np.nan

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_clean = df.copy()

        # Numeric conversion
        numeric_cols = ['user_nfollower', 'user_total_like', 'vid_nview', 'vid_nlike',
                        'vid_ncomment', 'vid_nshare', 'vid_nsave', 'music_nused']
        for col in numeric_cols:
            df_clean[col] = df_clean[col].apply(self.convert_units)

        # Text preprocessing
        df_clean['vid_desc_clean'] = df_clean['vid_caption'].fillna('')
        df_clean['vid_desc_clean'] = df_clean['vid_desc_clean'].apply(self.remove_emoji)
        df_clean['vid_desc_clean'] = df_clean['vid_desc_clean'].apply(self.remove_stopwords)

        # Hashtag normalization
        df_clean['vid_hashtags_normalized'] = df_clean['vid_hashtags'].apply(self.normalize_hashtags)
        df_clean['hashtag_count'] = df_clean['vid_hashtags_normalized'].apply(len)

        # Timestamp conversion
        df_clean['vid_postTime'] = df_clean['vid_postTime'].apply(self.convert_to_vietnam_time)
        df_clean['vid_scrapeTime'] = df_clean['vid_scrapeTime'].apply(self.convert_to_vietnam_time)

        # Duration conversion
        df_clean['vid_duration_sec'] = df_clean['vid_duration'].apply(self.duration_to_seconds)

        # Fill missing
        df_clean['vid_hashtags'] = df_clean['vid_hashtags'].fillna('')
        df_clean['music_title'] = df_clean['music_title'].fillna('Unknown')
        df_clean['music_authorName'] = df_clean['music_authorName'].fillna('Unknown')

        # Time-based features
        df_clean['vid_existtime_hrs'] = (df_clean['vid_scrapeTime'] - df_clean['vid_postTime']).dt.total_seconds() / 3600
        df_clean['post_hour'] = df_clean['vid_postTime'].dt.hour
        df_clean['post_day'] = df_clean['vid_postTime'].dt.day_name()

        df_clean.sort_values(['user_name', 'vid_id', 'vid_scrapeTime'], inplace=True)

        return df_clean
