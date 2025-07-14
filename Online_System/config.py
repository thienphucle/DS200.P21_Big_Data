# config.py

KAFKA_CONFIG = {
    "bootstrap_servers": "localhost:9092",
    "streaming_topic": "streaming_data_topic",
    "delay_between_messages": 1,
}

FEATURE_ENGINEERING_CONFIG = {
    "rolling_window_size": 5,  
    "snapshot_limit_per_video": 3,
    "max_caption_length": 256
}

DATA_PATHS = {
    "streaming_data_path": r"D:\UIT\DS200\DS200_Project\Dataset\Raw_Data\Merged_Data\streaming_data.csv",
    "output_feature_path": r"D:\UIT\DS200\DS200_Project\Results"
}

MODEL_PATH = {
    "best_model": r"D:\UIT\DS200\DS200_Project\ModelResults\enhanced_tiktok_model.pth",
}

DASHBOARD_CONFIG = {
    "online_prediction_topic" : "online_prediction_topic",
    "refresh_interval_seconds": 3
}

MONGODB_CONFIG = {
    "uri": "mongodb://localhost:27017",
    "database": "TikTokPrediction",
    "collection": "predictions"
}

SPARK_CONFIG = {
    "executor_memory": "1g",
    "driver_memory": "1g",
    "checkpoint_location": "./checkpoints"
}