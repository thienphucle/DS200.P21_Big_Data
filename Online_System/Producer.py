from kafka import KafkaProducer
import pandas as pd
import json
from time import sleep
from typing import List, Dict, Any
from datetime import datetime
from config import KAFKA_CONFIG, DATA_PATHS


class TikTokProducer:
    def __init__(self):
        self.topic = KAFKA_CONFIG['streaming_topic']
        self.bootstrap_servers = KAFKA_CONFIG['bootstrap_servers']
        self.delay = KAFKA_CONFIG['delay_between_messages']
        self.file_path = DATA_PATHS['streaming_data_path']

        self.producer = KafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            value_serializer=self._json_serializer
        )

    # chuyển data thành dạng json 
    def _json_serializer(self, data: Dict[str, Any]) -> bytes:
        return json.dumps(data, default=str).encode('utf-8')

    def _load_snapshots_from_csv(self) -> List[Dict[str, Any]]:
        df = pd.read_csv(self.file_path)

        df = df.sort_values(['user_name', 'vid_id', 'vid_scrapeTime'])

        df = df.groupby(['user_name', 'vid_id']).head(3).reset_index(drop=True)
        # print(f"====[PRODUCER] - Dataframe  sau khi nhom theo 'user_name' va 'vid_id':\n{df.head(3)}")

        return df.to_dict(orient='records')

    def send_snapshots(self, records: List[Dict[str, Any]]):

        print(f"Sending {len(records)} records to Kafka topic '{self.topic}'...")
        
        # Gửi từng bản ghi 1 cách tuần tự và chậm rãi
        for record in records:
            self.producer.send(self.topic, value=record)
            sleep(self.delay)

        # Đảm bảo toàn bộ dữ liệu được gửi
        self.producer.flush()
        print("All records sent.")

    def send_from_csv(self):
        records = self._load_snapshots_from_csv()
        self.send_snapshots(records)

    def send_snapshots_grouped(self):
        df = pd.read_csv(self.file_path)
    
        # Sắp xếp snapshot theo thời gian
        df = df.sort_values(['user_name', 'vid_id', 'vid_scrapeTime'])

        # Chỉ giữ 3 snapshot đầu tiên mỗi video
        grouped = df.groupby(['user_name', 'vid_id']).head(3).reset_index(drop=True)

        # Gộp theo video
        video_groups = grouped.groupby(['user_name', 'vid_id'])

        count = 0
        for (user, vid), group in video_groups:
            records = group.to_dict(orient="records")

            if len(records) < 3:
                continue  # Bỏ qua video chưa đủ 3 snapshot

            for record in records:
                # print(f"====[PRODUCER] - Send record for vid_id={vid} ====")
                self.producer.send(self.topic, value=record)
                sleep(self.delay)

            count += 1

        self.producer.flush()
        print(f"✅ Sent {count} videos with ≥3 snapshots to Kafka.")


if __name__ == "__main__":
    producer = TikTokProducer()
    # print("============================STARTING STREAMING DATA TO INPUT KAFKA TOPIC============================")
    producer.send_from_csv()
    # producer.send_from_csv()
