# streaming.py

from kafka import KafkaConsumer
import json
import pandas as pd
import joblib
from datetime import datetime
from collections import defaultdict
from kafka import KafkaProducer

from config import KAFKA_CONFIG, FEATURE_ENGINEERING_CONFIG, MODEL_PATH
from Online_System import Feature_Engineer
from Online_System.Feature_Engineer import TikTokFeatureEngineerOnline

class StreamProcessor:
    def __init__(self):
        self.topic = KAFKA_CONFIG['video_snapshot_topic'] # Kafka Topic 
        self.bootstrap_servers = KAFKA_CONFIG['bootstrap_servers'] # Bootstrap server
        self.snapshot_limit = FEATURE_ENGINEERING_CONFIG['snapshot_limit_per_video'] # Số lượng snapshot 
        self.window_size = FEATURE_ENGINEERING_CONFIG['rolling_window_size'] # Số video gần nhất để phân tích 
        self.model = MODEL_PATH['best_model'] # Best model 

        # Bộ nhớ tạm thời
        self.user_video_history = defaultdict(list)  # {user_name: [video1_dict, video2_dict, ...]}
        self.video_snapshots = defaultdict(list)     # {vid_id: [snapshot1, snapshot2, ...]}

        # Load model đã train
        self.model = joblib.load(self.model)

        # Khởi tạo Kafka consumer
        self.consumer = KafkaConsumer(
            self.topic,
            bootstrap_servers=self.bootstrap_servers,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            auto_offset_reset='latest',
            enable_auto_commit=True,
            group_id='video_predictor_group'
        )

        self.producer = KafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )

        self.fe = TikTokFeatureEngineerOnline(n_recent_videos=self.window_size)

    def process_snapshot(self, snapshot: dict):
        vid_id = snapshot['vid_id']
        user_name = snapshot['user_name']

        self.video_snapshots[vid_id].append(snapshot)

        # Nếu đã đủ số lượng snapshot cho một video thì xử lý
        if len(self.video_snapshots[vid_id]) == self.snapshot_limit:
            video_df = pd.DataFrame(self.video_snapshots[vid_id])
            video_df['vid_postTime'] = pd.to_datetime(video_df['vid_postTime'])
            video_df['vid_scrapeTime'] = pd.to_datetime(video_df['vid_scrapeTime'])

            # Sort lại để đảm bảo đúng thứ tự snapshot
            video_df = video_df.sort_values(by="vid_scrapeTime")

            # Trích xuất đặc trưng
            feature_row = self.fe._extract_video_features(video_df)

            # Thêm thông tin lịch sử video của user
            self.user_video_history[user_name].append(feature_row)
            if len(self.user_video_history[user_name]) > self.window_size:
                self.user_video_history[user_name].pop(0)

            # Trích xuất đặc trưng xu hướng từ lịch sử video
            df_history = pd.DataFrame(self.user_video_history[user_name])
            full_feature = self.fe._extract_user_trend_features(df_history)

            # Gộp với feature video hiện tại
            final_input = pd.concat([feature_row.reset_index(drop=True), full_feature.reset_index(drop=True)], axis=1)

            # Gọi model dự đoán
            prediction = self.model.predict(final_input)[0]
            print(f"[PREDICTED] vid_id: {vid_id}, predicted_next_interaction: {prediction:.2f}")

            # Gửi kết quả sang Kafka topic 'video_predictions'
            prediction_record = {
                'vid_id': vid_id,
                'user_name': user_name,
                'prediction': round(prediction, 2),
                'timestamp': datetime.now().isoformat()
            }
            self.producer.send('video_predictions', value=prediction_record)
            self.producer.flush()


            # Xoá snapshot khỏi bộ nhớ tạm
            del self.video_snapshots[vid_id]

    def start_stream(self):
        print(f"Listening to Kafka topic: {self.topic}")
        for message in self.consumer:
            snapshot = message.value
            try:
                self.process_snapshot(snapshot)
            except Exception as e:
                print(f"[ERROR] Processing snapshot failed: {e}")


if __name__ == "__main__":
    stream_processor = StreamProcessor()
    stream_processor.start_stream()