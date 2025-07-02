# Prediction.py
import sys
import os
os.environ['PYSPARK_PYTHON'] = r'C:\Users\duong\anaconda3\python.exe'

# Thêm thư mục cha vào sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import pandas as pd
import json
from datetime import datetime

from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, struct, collect_list, size
from pyspark.sql.types import *

from config import KAFKA_CONFIG, FEATURE_ENGINEERING_CONFIG, MODEL_PATH, DASHBOARD_CONFIG
from Feature_Engineer import TikTokFeatureEngineerOnline
from Offline_System.model import TikTokGrowthPredictor, TikTokDataset


import logging
logging.getLogger("py4j").setLevel(logging.ERROR)
logging.getLogger("pyspark").setLevel(logging.INFO)
import traceback

# Giữ mô hình và feature engineer dưới dạng biến toàn cục (singleton) để tránh load lại nhiều lần trong Spark worker.
_model = None
_fe = None

def get_model_and_fe():
    global _model, _fe
    if _model is None or _fe is None:
        from Feature_Engineer import TikTokFeatureEngineerOnline
        from Offline_System.model import TikTokGrowthPredictor, TikTokDataset

        device = 'cpu'
        model_data = torch.load(MODEL_PATH["best_model"], map_location=device)
        model = TikTokGrowthPredictor(text_dim=model_data["model_config"]["text_dim"])
        model.load_state_dict(model_data["model_state_dict"])
        model.to(device)
        model.eval() # Chuyển về chế độ .eval() để inference.

        fe = TikTokFeatureEngineerOnline(
            n_recent_videos=FEATURE_ENGINEERING_CONFIG['rolling_window_size'],
            min_snapshots=FEATURE_ENGINEERING_CONFIG['snapshot_limit_per_video']
        )

        _model = model
        _fe = fe

    return _model, _fe

def predict_from_snapshots(iterator):
    import pandas as pd
    import torch
    from datetime import datetime
    from Offline_System.model import TikTokDataset

    model, fe = get_model_and_fe()
    device = 'cpu'

    for pdf in iterator:
        try:
            results = []

            for idx in range(len(pdf)):
                snapshots = pdf.iloc[idx]['snapshots']
                if not isinstance(snapshots, list) or len(snapshots) == 0:
                    continue

                snapshots_df = pd.DataFrame(snapshots)
                snapshots_df["vid_postTime"] = pd.to_datetime(snapshots_df["vid_postTime"], errors='coerce')
                snapshots_df["vid_scrapeTime"] = pd.to_datetime(snapshots_df["vid_scrapeTime"], errors='coerce')

                df_infer = fe.transform(snapshots_df)
                if df_infer.get("inference_data") is None or df_infer["inference_data"].empty:
                    continue

                dataset = TikTokDataset(
                    df_infer['inference_data'],
                    df_infer['user_trend_features'],
                    df_infer['text_features'],
                    mode='predict'
                )
                if len(dataset) == 0:
                    print("Dataset is empty!")
                    continue
               
                sample = dataset[0]
                text = sample["text_features"].unsqueeze(0).to(device)
                struct = sample["structured_features"].unsqueeze(0).to(device)
                time = sample["time_features"].unsqueeze(0).to(device)

                with torch.no_grad():
                    reg_output, _, _ = model(text, struct, time)
                    pred = reg_output[0][0].item()

                results.append({
                    "user_name": snapshots_df["user_name"].iloc[0],
                    "vid_id": snapshots_df["vid_id"].iloc[0],
                    "prediction": round(float(pred), 2),
                    "timestamp": datetime.utcnow().isoformat()
                })

            if results:
                df_result = pd.DataFrame(results)
                df_result["prediction"] = df_result["prediction"].astype(float) 
                yield df_result
                
            else:
                yield pd.DataFrame([], columns=["user_name", "vid_id", "prediction", "timestamp"])
        except Exception as e:
            print("[Prediction Error]", e)
            traceback.print_exc()
            yield pd.DataFrame([], columns=["user_name", "vid_id", "prediction", "timestamp"])

class TikTokProcessor:
    def __init__(self):
        self.bootstrap_servers = KAFKA_CONFIG['bootstrap_servers']
        self.input_topic = KAFKA_CONFIG['streaming_topic']
        self.output_topic = DASHBOARD_CONFIG['online_prediction_topic']
        self.min_snapshots = FEATURE_ENGINEERING_CONFIG['snapshot_limit_per_video']

        self.schema = self._define_schema()

        scala_version ='2.12'
        spark_version ='3.5.5'

        packages = [
            f'org.apache.spark:spark-sql-kafka-0-10_{scala_version}:{spark_version}',
            'org.apache.kafka:kafka-clients:3.6.0']

        self.spark = SparkSession.builder.master("local[*]")\
            .appName("kafka-example")\
            .config("spark.jars.packages", ",".join(packages))\
            .config("spark.executor.memory", "4g")\
            .config("spark.driver.memory", "4g")\
            .config("spark.sql.shuffle.partitions", "4")\
            .getOrCreate()
        # Tắt Arrow để tránh lỗi liên quan mapInPandas.
        self.spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "false")
        self.spark.conf.set("spark.sql.adaptive.enabled", "false")
        # self.spark.sparkContext.setLogLevel("DEBUG")

    # Định nghĩa schema của message Kafka.
    def _define_schema(self):
        return StructType([
            StructField("user_name", StringType()),
            StructField("user_nfollower", FloatType()),
            StructField("user_total_likes", FloatType()),
            StructField("vid_id", StringType()),
            StructField("vid_caption", StringType()),
            StructField("vid_postTime", TimestampType()),
            StructField("vid_scrapeTime", TimestampType()),
            StructField("vid_duration", FloatType()),
            StructField("vid_nview", FloatType()),
            StructField("vid_nlike", FloatType()),
            StructField("vid_ncomment", FloatType()),
            StructField("vid_nshare", FloatType()),
            StructField("vid_nsave", FloatType()),
            StructField("vid_hashtags", StringType()),
            StructField("vid_url", StringType()),
            StructField("music_id", StringType()),
            StructField("music_title", StringType()),
            StructField("music_nused", StringType()),
            StructField("music_authorName", StringType()),
            StructField("music_originality", StringType()),
            StructField("topic", StringType()),
            StructField("vid_desc_clean", StringType()),
            StructField("vid_hashtags_normalized", StringType()),
            StructField("hashtag_count", IntegerType()),
            StructField("vid_duration_sec", FloatType()),
            StructField("vid_existtime_hrs", FloatType()),
            StructField("post_hour", StringType()),
            StructField("post_day", StringType()),
        ])

    def start(self):
        # Đọc stream Kafka từ topic đầu vào.
        df_raw = self.spark.readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", self.bootstrap_servers) \
            .option("subscribe", self.input_topic) \
            .option("startingOffsets", "latest") \
            .load()
        
        # Parse JSON message từ Kafka thành các cột cụ thể theo schema
        df_parsed = df_raw.select(from_json(col("value").cast("string"), self.schema).alias("data")).select("data.*")
        
        # Nhóm các bản snapshot lại theo video ID.
        # Chỉ lấy những video có đủ snapshot theo cấu hình (snapshot_limit_per_video).
        df_grouped = df_parsed \
            .withWatermark("vid_scrapeTime", "10 minutes") \
            .groupBy("user_name", "vid_id") \
            .agg(collect_list(struct("*")).alias("snapshots")) \
            .filter(size(col("snapshots")) >= self.min_snapshots)
        
        # Schema đầu ra chứa các prediction.
        output_schema = StructType([
            StructField("user_name", StringType(), True),
            StructField("vid_id", StringType(), True),
            StructField("prediction", DoubleType(), True),
            StructField("timestamp", StringType(), True)
        ])

        # Dùng mapInPandas() để áp dụng hàm dự đoán batch-wise.
        df_predicted = df_grouped.mapInPandas(predict_from_snapshots, schema=output_schema)

        # Ghi kết quả prediction về Kafka output topic.
        df_output = df_predicted.selectExpr(
            "vid_id as key",
            """to_json(named_struct(
                'user_name', user_name,
                'vid_id', vid_id,
                'prediction', prediction,
                'timestamp', timestamp
            )) as value"""
        )

        query = df_output.writeStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", self.bootstrap_servers) \
            .option("topic", self.output_topic) \
            .option("checkpointLocation", "./spark_checkpoint") \
            .outputMode("update") \
            .start()

        print(f"✅ Spark stream started: {self.input_topic} → {self.output_topic}")
    
        query.awaitTermination()


if __name__ == "__main__":
    processor = TikTokProcessor()
    processor.start()
    
    print("✅ Spark query started")