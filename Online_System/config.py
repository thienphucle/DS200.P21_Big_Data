# config.py

KAFKA_CONFIG = {
    "bootstrap_servers": "localhost:9092",
    "video_snapshot_topic": "video_snapshots",
    "delay_between_messages": 2,  # seconds
}

FEATURE_ENGINEERING_CONFIG = {
    "rolling_window_size": 20,  # số video gần nhất để tính trung bình
    "snapshot_limit_per_video": 4,
    "max_caption_length": 256
}

DATA_PATHS = {
    "snapshot_csv_path": "snapshots.csv",
    "output_feature_path": "features_output/"
}

MODEL_PATH = {
    "best_model": " ",
}
