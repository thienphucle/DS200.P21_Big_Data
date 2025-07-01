# config.py

KAFKA_CONFIG = {
    "bootstrap_servers": "localhost:9092",
    "streaming_topic": "streaming_data_topic",
    "delay_between_messages": 2,  # seconds
}

FEATURE_ENGINEERING_CONFIG = {
    "rolling_window_size": 20,  # số video gần nhất để tính trung bình
    "snapshot_limit_per_video": 4,
    "max_caption_length": 256
}

DATA_PATHS = {
    "streaming_data_path": r"D:\UIT\DS200\DS200_Project\Dataset\Preprocessed_Data\streaming_data.csv",
    "output_feature_path": r"D:\UIT\DS200\DS200_Project\Results"
}

MODEL_PATH = {
    "best_model": r"D:\UIT\DS200\DS200_Project\Results\Best_Model\tiktok_model.pkl",
}

DASHBOARD_CONFIG = {
    "online_prediction_topic" : "online_prediction_topic",
}