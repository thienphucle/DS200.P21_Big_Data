import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

KAFKA_CONFIG = {
    "bootstrap.servers": "localhost:9092",
    "input_topic": "input",
    "output_topic": "output"
}

SPARK_CONFIG = {
    "executor_memory": "8g",
    "driver_memory": "4g",
    "checkpoint_location": os.path.join(BASE_DIR, "checkpoint"),
    "output_path": "/home/trungtran0165/Downloads/IE212.P11/data/output"
}

MODEL_CONFIG = {
    "model_path": "models/bert_multilabel_model_trainer",  # Đường dẫn tới thư mục đã giải nén
    "preprocessor_path": "models/vietnamese_preprocessor.pkl",
    "label_mapping_path": "models/label_mapping.pkl"
}

