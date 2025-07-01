from kafka import KafkaConsumer, KafkaProducer
import json
import pandas as pd
import joblib
from datetime import datetime
from collections import defaultdict

from config import KAFKA_CONFIG, FEATURE_ENGINEERING_CONFIG, MODEL_PATH, DASHBOARD_CONFIG
from Feature_Engineer import TikTokFeatureEngineerOnline


class StreamProcessor:
    def __init__(self):
        # Kafka config
        self.topic = KAFKA_CONFIG['streaming_topic']
        self.bootstrap_servers = KAFKA_CONFIG['bootstrap_servers']
        self.predict_topic = DASHBOARD_CONFIG['online_prediction_topic']

        # Feature config
        self.snapshot_limit = FEATURE_ENGINEERING_CONFIG['snapshot_limit_per_video']
        self.window_size = FEATURE_ENGINEERING_CONFIG['rolling_window_size']

        # Load pretrained model
        self.model = joblib.load(MODEL_PATH['best_model'])

        # Kafka consumer
        self.consumer = KafkaConsumer(
            self.topic,
            bootstrap_servers=self.bootstrap_servers,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            auto_offset_reset='latest',
            enable_auto_commit=True,
            group_id='video_predictor_group'
        )

        # Kafka producer
        self.producer = KafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )

        # Feature engineer instance
        self.fe = TikTokFeatureEngineerOnline(n_recent=self.window_size, min_snapshots=self.snapshot_limit)


        self.video_snapshots = defaultdict(list)       # {vid_id: [snapshot1, snapshot2, ...]}
        self.user_video_history = defaultdict(list)    # {user_name: [training_row_dicts]}

    def process_snapshot(self, snapshot: dict):
        vid_id = snapshot['vid_id']
        user_name = snapshot['user_name']

        self.video_snapshots[vid_id].append(snapshot)

        if len(self.video_snapshots[vid_id]) < self.snapshot_limit:
            return  # wait for more snapshots

        # Convert to DataFrame
        df_snapshots = pd.DataFrame(self.video_snapshots[vid_id])
        df_snapshots['vid_postTime'] = pd.to_datetime(df_snapshots['vid_postTime'])
        df_snapshots['vid_scrapeTime'] = pd.to_datetime(df_snapshots['vid_scrapeTime'])

        # Step 1: Preprocess raw data
        df_preprocessed = self.fe._preprocess_raw_data(df_snapshots)

        # Step 2: Extract video-level snapshot delta features
        df_video_features = self.fe._extract_video_level_features(df_preprocessed)
        if df_video_features.empty:
            print(f"[WARN] Not enough snapshots for vid_id: {vid_id}")
            return

        # Step 3: Create inference row (uses first 2 snapshots to predict 3rd)
        df_inference = self.fe._create_inference_data(df_video_features)
        if df_inference.empty:
            print(f"[WARN] Not enough valid snapshots for inference vid_id: {vid_id}")
            return

        # Step 4: Update user history
        self.user_video_history[user_name].append(df_inference.iloc[0])

        if len(self.user_video_history[user_name]) > self.window_size:
            self.user_video_history[user_name].pop(0)

        # Step 5: Create user-level trend features
        user_history_df = pd.DataFrame(self.user_video_history[user_name])
        df_user_trend = self.fe._extract_user_trend_features(user_history_df)

        # Step 6: Combine video inference and user trend
        final_input = pd.concat([df_inference.reset_index(drop=True), df_user_trend.reset_index(drop=True)], axis=1)

        # Step 7: Predict
        prediction = self.model.predict(final_input)[0]
        print(f"[✅ PREDICTED] vid_id: {vid_id} → predicted_next_engagement: {prediction:.2f}")

        # Step 8: Send to Kafka
        prediction_message = {
            'vid_id': vid_id,
            'user_name': user_name,
            'prediction': round(prediction, 2),
            'timestamp': datetime.now().isoformat()
        }

        self.producer.send(self.predict_topic, value=prediction_message)
        self.producer.flush()

        # Cleanup memory
        del self.video_snapshots[vid_id]

    def start_stream(self):
        print(f"🔄 Listening to Kafka topic: {self.topic}")
        for message in self.consumer:
            snapshot = message.value
            try:
                self.process_snapshot(snapshot)
            except Exception as e:
                print(f"[❌ ERROR] Failed to process snapshot: {e}")


if __name__ == "__main__":
    stream_processor = StreamProcessor()
    stream_processor.start_stream()
