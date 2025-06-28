# producer.py

from kafka import KafkaProducer
import pandas as pd
import json
from time import sleep
from typing import List, Dict, Any

from config import KAFKA_CONFIG, DATA_PATHS


class SnapshotProducer:
    def __init__(self):
        self.topic = KAFKA_CONFIG['video_snapshot_topic']
        self.bootstrap_servers = KAFKA_CONFIG['bootstrap_servers']
        self.delay = KAFKA_CONFIG['delay_between_messages']

        self.producer = KafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            value_serializer=self._json_serializer
        )

    def _json_serializer(self, data: Dict[str, Any]) -> bytes:
        return json.dumps(data, default=str).encode('utf-8')

    def _load_snapshots_from_csv(self, filepath: str) -> List[Dict[str, Any]]:
        df = pd.read_csv(filepath)
        df['vid_postTime'] = pd.to_datetime(df['vid_postTime']).astype(str)
        df['vid_scrapeTime'] = pd.to_datetime(df['vid_scrapeTime']).astype(str)
        return df.to_dict(orient='records')

    def send_snapshots(self, records: List[Dict[str, Any]]):
        print(f"Sending {len(records)} records to Kafka topic '{self.topic}'...")
        for record in records:
            self.producer.send(self.topic, value=record)
            sleep(self.delay)
        self.producer.flush()
        print("All records sent.")

    def send_from_csv(self, filepath: str = None):
        filepath = filepath or DATA_PATHS['snapshot_csv_path']
        records = self._load_snapshots_from_csv(filepath)
        self.send_snapshots(records)


if __name__ == "__main__":
    producer = SnapshotProducer()
    producer.send_from_csv()
