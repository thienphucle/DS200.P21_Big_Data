import sys
import os
import gc
import logging
from typing import Iterator, Dict, Any, Optional

# os.environ['PYSPARK_PYTHON'] = r'C:\\Users\\duong\\anaconda3\\python.exe'
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYTHONPATH'] = os.pathsep.join([
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..')),
    os.environ.get('PYTHONPATH', '')
])
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import torch
import pandas as pd
import numpy as np
from datetime import datetime
import traceback

from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, struct, collect_list, size
from pyspark.sql.types import *
from pyspark.sql.functions import to_timestamp
from pyspark.sql.functions import to_json, expr


from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import TopicAlreadyExistsError
import logging 


# Enhanced imports with error handling
try:
    from config import KAFKA_CONFIG, FEATURE_ENGINEERING_CONFIG, MODEL_PATH, DASHBOARD_CONFIG
    from Feature_Engineer import TikTokFeatureEngineerOnline
    from Offline_System.Bmodel import EnhancedTikTokGrowthPredictor, TikTokDataset
    from Preprocessor import TikTokPreprocessor
except ImportError as e:
    logger.error(f"Import error: {e}")
    raise

_model = None
_fe = None
_model_config = None
_pre = TikTokPreprocessor() 

column_names = ['user_name', 'user_nfollower', 'user_total_likes', 'vid_id', 'vid_caption', 'vid_postTime', 'vid_scrapeTime', 'vid_duration', 'vid_nview', 'vid_nlike', 'vid_ncomment', 'vid_nshare', 'vid_nsave', 'vid_hashtags', 'vid_url', 'music_id', 'music_title', 'music_nused', 'music_authorName', 'music_originality', 'topic']

def get_model_and_fe():
    global _model, _fe, _model_config
    
    if _model is None or _fe is None:
        try:
            logger.info("Loading model and feature engineer...")
            device = 'cpu'
            
            if not os.path.exists(MODEL_PATH["best_model"]):
                raise FileNotFoundError(f"Model file not found: {MODEL_PATH['best_model']}")
            
            model_data = torch.load(MODEL_PATH["best_model"], map_location=device)
            _model_config = model_data.get("model_config", {})
            
            required_keys = ['text_dim', 'structured_dim', 'temporal_dim']
            for key in required_keys:
                if key not in _model_config:
                    logger.warning(f"Missing {key} in model config, using default")
                    _model_config[key] = 512 if key == 'text_dim' else 64
            
            model = EnhancedTikTokGrowthPredictor(
                text_dim=_model_config["text_dim"],
                structured_dim=_model_config["structured_dim"],
                temporal_dim=_model_config["temporal_dim"],
                fusion_dim=512,
                num_regression_outputs=5,
                num_classification_outputs=6,
                num_classes=3
            )

            # Load model và in ra lỗi nếu xuất hiện lỗi 
            try:
                model.load_state_dict(model_data["model_state_dict"])
            except Exception as e:
                logger.error(f"Error loading model state: {e}")
                raise
            
            model.to(device)
            model.eval()
            
            fe = TikTokFeatureEngineerOnline(
                n_recent_videos=FEATURE_ENGINEERING_CONFIG.get('rolling_window_size', 5),
                min_snapshots=FEATURE_ENGINEERING_CONFIG.get('snapshot_limit_per_video', 3)
            )
            
            _model = model
            _fe = fe
            
            logger.info("Model and feature engineer loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            traceback.print_exc()
            raise
    
    return _model, _fe

def predict_from_snapshots(iterator: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
    model, fe  = get_model_and_fe()
    device = 'cpu'

    for pdf in iterator:
        try:
            results = []

            for idx in range(len(pdf)):
                try:
                    snapshots = pdf.iloc[idx]['snapshots']
                    if not isinstance(snapshots, list) or len(snapshots) == 0:
                        continue
                    snapshots_df = pd.DataFrame(snapshots)
                    # print(f"====[PREDICTION] - cac cot cua snapshot:\n{snapshots_df.columns}")
                    # print(f"====[PREDICTION] - snapshots_df:\n{snapshots_df}")

                    snapshots_df = _pre.transform(snapshots_df)
                    snapshots_df.dropna(subset=["vid_postTime", "vid_scrapeTime"], inplace=True)
                    if snapshots_df.empty:
                        continue
                    
                    input_fe = fe.transform(snapshots_df)
                    inference_df = input_fe.get("training_data")
                    if inference_df is None or inference_df.empty:
                        continue

                    user = str(snapshots_df["user_name"].iloc[0])
                    vid = str(snapshots_df["vid_id"].iloc[0])

                    row_match = inference_df[
                        (inference_df["user_name"].astype(str) == user) &
                        (inference_df["vid_id"].astype(str) == vid)
                    ]
                    if row_match.empty:
                        continue

                    match_idx = row_match.index[0]
                    dataset = TikTokDataset(
                        inference_df,
                        input_fe["user_trend_features"],
                        input_fe["text_features"],
                        mode='predict'
                    )
                    if len(dataset) == 0:
                        continue

                    dataset_idx = min(match_idx, len(dataset) - 1)
                    sample = dataset[dataset_idx]

                    text = sample["text_features"].unsqueeze(0).to(device).float()
                    struct_feat = sample["structured_features"].unsqueeze(0).to(device).float()
                    time_feat = sample["time_features"].unsqueeze(0).to(device).float()

                    with torch.no_grad():
                        reg_output, _, cls_output, _ = model(text, struct_feat, time_feat)

                        predictions = reg_output[0].tolist()  # list of 5 float
                        cls_probs = torch.softmax(cls_output[0, -1, :], dim=0)
                        cls_label = torch.argmax(cls_probs).item()
                        cls_label_str = ["decrease", "stable", "increase"][cls_label]
                        confidence = float(torch.max(cls_probs).item())

                    result = {
                        "user_name": user,
                        "vid_id": vid,
                        "timestamp": datetime.utcnow().isoformat(),
                        "prediction_view": float(predictions[0]),
                        "prediction_like": float(predictions[1]),
                        "prediction_comment": float(predictions[2]),
                        "prediction_share": float(predictions[3]),
                        "prediction_save": float(predictions[4]),
                        "growth_category": cls_label_str,
                        "confidence": round(confidence, 4)
                    }

                    results.append(result)

                except Exception as e:
                    logger.error(f"[FATAL ROW ERROR] idx={idx}, exception={e}")
                    traceback.print_exc()
                    continue

            yield pd.DataFrame(results) if results else pd.DataFrame([], columns=[
                "user_name", "vid_id", "timestamp",
                "prediction_view", "prediction_like", "prediction_comment",
                "prediction_share", "prediction_save",
                "growth_category", "confidence"
            ])
            
        except Exception as e:
            logger.error(f"Batch-level error: {e}")
            traceback.print_exc()
            yield pd.DataFrame([], columns=[
                "user_name", "vid_id", "timestamp",
                "prediction_view", "prediction_like", "prediction_comment",
                "prediction_share", "prediction_save",
                "growth_category", "confidence"
            ])
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

class TikTokProcessor:
    def __init__(self):
        self.bootstrap_servers = KAFKA_CONFIG['bootstrap_servers']
        self.input_topic = KAFKA_CONFIG['streaming_topic']
        self.output_topic = DASHBOARD_CONFIG['online_prediction_topic']
        self.min_snapshots = FEATURE_ENGINEERING_CONFIG.get('snapshot_limit_per_video', 3)
        self.schema = self._define_schema()

        # Enhanced Spark configuration
        scala_version = '2.12'
        spark_version = '3.5.5'
        packages = [
            f'org.apache.spark:spark-sql-kafka-0-10_{scala_version}:{spark_version}',
            'org.apache.kafka:kafka-clients:3.6.0'
        ]

        self.spark = SparkSession.builder.master("local[*]")\
            .appName("enhanced-tiktok-prediction")\
            .config("spark.jars.packages", ",".join(packages))\
            .config("spark.executor.memory", "1g")\
            .config("spark.driver.memory", "1g")\
            .config("spark.executor.cores", "2")\
            .config("spark.sql.shuffle.partitions", "2")\
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")\
            .config("spark.sql.execution.arrow.pyspark.enabled", "false")\
            .config("spark.sql.adaptive.enabled", "false")\
            .config("spark.sql.adaptive.coalescePartitions.enabled", "false")\
            .config("spark.executor.heartbeatInterval", "60s")\
            .config("spark.network.timeout", "300s")\
            .config("spark.sql.streaming.checkpointLocation.deleteOnStop", "false")\
            .config("spark.python.worker.memory", "4g")\
            .config("spark.executor.pyspark.memory", "4g")\
            .getOrCreate()

        self.spark.sparkContext.setLogLevel("WARN")

        # Nếu cần giảm cả log của hệ thống (như Ivy)
        logging.getLogger("py4j").setLevel(logging.ERROR)

        logger.info("Enhanced Spark session created successfully")

    def _define_schema(self):
        """Define enhanced schema for Kafka messages"""
        return StructType([StructField("user_name", StringType(), True),
                             StructField("user_nfollower", StringType(), True),
                             StructField("user_total_likes", StringType(), True),
                             StructField("vid_id", StringType(), True),
                             StructField("vid_caption", StringType(), True),
                             StructField("vid_postTime", StringType(), True),
                             StructField("vid_scrapeTime", StringType(), True),
                             StructField("vid_duration", StringType(), True),
                             StructField("vid_nview", StringType(), True),
                             StructField("vid_nlike", StringType(), True),
                             StructField("vid_ncomment", StringType(), True),
                             StructField("vid_nshare", StringType(), True),
                             StructField("vid_nsave", StringType(), True),
                             StructField("vid_hashtags", StringType(), True),
                             StructField("vid_url", StringType(), True),
                             StructField("music_id", StringType(), True),
                             StructField("music_title", StringType(), True),
                             StructField("music_nused", StringType(), True),
                             StructField("music_authorName", StringType(), True),
                             StructField("music_originality", StringType(), True),
                             StructField("topic", StringType(), True)])

    def _ensure_topic_exists(self, topic_name: str):
        try:
            admin_client = KafkaAdminClient(bootstrap_servers=self.bootstrap_servers)
            existing_topics = admin_client.list_topics()

            if topic_name not in existing_topics:
                topic = NewTopic(name=topic_name, num_partitions=1, replication_factor=1)
                admin_client.create_topics([topic])
                logger.info(f"✅ Created topic '{topic_name}'")
            else:
                logger.info(f"✔️ Topic '{topic_name}' already exists")
        except TopicAlreadyExistsError:
            logger.info(f"✔️ Topic '{topic_name}' already exists")
        except Exception as e:
            logger.error(f"Error ensuring topic exists: {e}")
        finally:
            admin_client.close()

    def start(self):
        try:
            logger.info("Starting enhanced streaming pipeline...")

            df_raw = self.spark.readStream \
                .format("kafka") \
                .option("kafka.bootstrap.servers", self.bootstrap_servers) \
                .option("subscribe", self.input_topic) \
                .option("startingOffsets", "latest") \
                .load()

            df_parsed = df_raw \
                .select(from_json(col("value").cast("string"), self.schema).alias("data")) \
                .select("data.*") \
                .withColumn("vid_scrapeTime", to_timestamp("vid_scrapeTime")) \
                .withColumn("vid_postTime", to_timestamp("vid_postTime"))

            df_grouped = df_parsed \
                .withWatermark("vid_scrapeTime", "15 minutes") \
                .groupBy("user_name", "vid_id") \
                .agg(collect_list(struct([col(c) for c in df_parsed.columns])).alias("snapshots")) \
                .filter(size(col("snapshots")) >= self.min_snapshots)

            self._ensure_topic_exists(self.output_topic)

            def predict_batch(df_batch, epoch_id):
                try:
                    pdf = df_batch.toPandas()
                    model, fe = get_model_and_fe()
                    device = 'cpu'
                    results = []

                    for idx, row in pdf.iterrows():
                        try:
                            snapshots = row["snapshots"]
                            if not isinstance(snapshots, list) or len(snapshots) == 0:
                                continue

                            snapshots_df = pd.DataFrame(snapshots)
                            
                            if snapshots_df.shape[1] == len(column_names):
                                snapshots_df.columns = column_names
                            else:
                                logger.warning(f"[ROW {idx}] ⚠️ Không khớp số lượng cột! {snapshots_df.shape[1]} cột thay vì {len(column_names)}.")
                                logger.warning(f"[ROW {idx}] snapshot_df.columns: {snapshots_df.columns.tolist()}")
                                continue  # Bỏ qua row không hợp lệ

                            # print(f"====[PREDICTION] - cac cot cua snapshot:\n{snapshots_df.columns}")
                            # print(f"====[PREDICTION] - snapshots_df:\n{snapshots_df}")

                            snapshots_df = _pre.transform(snapshots_df)
                            snapshots_df.dropna(subset=["vid_postTime", "vid_scrapeTime"], inplace=True)
                            if snapshots_df.empty:
                                continue

                            input_fe = fe.transform(snapshots_df)
                            inference_df = input_fe.get("training_data")
                            if inference_df is None or inference_df.empty:
                                continue

                            user = str(snapshots_df["user_name"].iloc[0])
                            vid = str(snapshots_df["vid_id"].iloc[0])

                            row_match = inference_df[
                                (inference_df["user_name"].astype(str) == user) &
                                (inference_df["vid_id"].astype(str) == vid)
                            ]
                            if row_match.empty:
                                continue

                            match_idx = row_match.index[0]
                            dataset = TikTokDataset(
                                inference_df,
                                input_fe["user_trend_features"],
                                input_fe["text_features"],
                                mode='predict'
                            )
                            if len(dataset) == 0:
                                continue

                            dataset_idx = min(match_idx, len(dataset) - 1)
                            sample = dataset[dataset_idx]

                            text = sample["text_features"].unsqueeze(0).to(device).float()
                            struct_feat = sample["structured_features"].unsqueeze(0).to(device).float()
                            time_feat = sample["time_features"].unsqueeze(0).to(device).float()

                            with torch.no_grad():
                                reg_output, _, cls_output, _ = model(text, struct_feat, time_feat)

                                predictions = reg_output[0].tolist()
                                cls_probs = torch.softmax(cls_output[0, -1, :], dim=0)
                                cls_label = torch.argmax(cls_probs).item()
                                cls_label_str = ["decrease", "stable", "increase"][cls_label]
                                confidence = float(torch.max(cls_probs).item())

                            result = {
                                "user_name": user,
                                "vid_id": vid,
                                "timestamp": datetime.utcnow().isoformat(),
                                "prediction_view": float(predictions[0]),
                                "prediction_like": float(predictions[1]),
                                "prediction_comment": float(predictions[2]),
                                "prediction_share": float(predictions[3]),
                                "prediction_save": float(predictions[4]),
                                "growth_category": cls_label_str,
                                "confidence": round(confidence, 4)
                            }
                            results.append(result)

                        except Exception as e:
                            logger.error(f"Error in row {idx}: {e}")
                            traceback.print_exc()

                    if results:
                        out_df = self.spark.createDataFrame(pd.DataFrame(results))
                        out_df = out_df.selectExpr(
                            "CAST(vid_id AS STRING) as key",
                            "to_json(struct(*)) as value"
                        )
                        out_df.write \
                            .format("kafka") \
                            .option("kafka.bootstrap.servers", self.bootstrap_servers) \
                            .option("topic", self.output_topic) \
                            .save()

                except Exception as e:
                    logger.error(f"Error in predict_batch: {e}")
                    traceback.print_exc()

            df_grouped.writeStream \
                .foreachBatch(predict_batch) \
                .outputMode("update") \
                .option("checkpointLocation", "./enhanced_spark_checkpoint") \
                .start() \
                .awaitTermination()

        except KeyboardInterrupt:
            logger.info("Process interrupted by user")
        except Exception as e:
            logger.error(f"Streaming pipeline error: {e}")
            traceback.print_exc()
        finally:
            logger.info("Prediction system shutdown complete")


if __name__ == "__main__":

    processor = TikTokProcessor()
    # print("===============STARTING READING DATA FROM INPUT KAFKA===============")
    processor.start()