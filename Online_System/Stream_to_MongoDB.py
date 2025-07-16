from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StringType, DoubleType
from config import KAFKA_CONFIG, DASHBOARD_CONFIG, MONGODB_CONFIG
import logging


class MongoStreamer:
    def __init__(self):
        self.kafka_topic = DASHBOARD_CONFIG['online_prediction_topic']
        self.bootstrap_servers = KAFKA_CONFIG['bootstrap_servers']
        self.mongo_uri = MONGODB_CONFIG['uri']
        self.mongo_db = MONGODB_CONFIG['database']
        self.mongo_collection = MONGODB_CONFIG['collection']

        scala_version = '2.12'
        spark_version = '3.5.5'
        packages = [
            f'org.apache.spark:spark-sql-kafka-0-10_{scala_version}:{spark_version}',
            'org.apache.kafka:kafka-clients:3.6.0',
            'org.mongodb.spark:mongo-spark-connector_2.12:10.5.0'
        ]

        self.spark = SparkSession.builder.master("local[*]") \
            .appName("KafkaToMongo") \
            .config("spark.jars.packages", ",".join(packages)) \
            .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
            .config("spark.sql.adaptive.enabled", "false") \
            .config("spark.mongodb.write.connection.uri", self.mongo_uri) \
            .getOrCreate()

        self.spark.sparkContext.setLogLevel("WARN")

        self.schema = StructType() \
            .add("user_name", StringType()) \
            .add("vid_id", StringType()) \
            .add("timestamp", StringType()) \
            .add("prediction_view", DoubleType()) \
            .add("prediction_like", DoubleType()) \
            .add("prediction_comment", DoubleType()) \
            .add("prediction_share", DoubleType()) \
            .add("prediction_save", DoubleType()) \
            .add("growth_category", StringType()) \
            .add("confidence", DoubleType())

    def write_batch_to_mongo(self, df, epoch_id):
        try:
            if df.isEmpty():
                print(f"[Epoch {epoch_id}] ❕ Empty batch")
                return

            df_clean = df.dropna()
            df_clean.write \
                .format("mongodb") \
                .mode("append") \
                .option("uri", self.mongo_uri) \
                .option("database", self.mongo_db) \
                .option("collection", self.mongo_collection) \
                .save()

            print(f"[Epoch {epoch_id}] ✅ Wrote {df_clean.count()} records to MongoDB")

        except Exception as e:
            print(f"[Epoch {epoch_id}] ❌ Error writing to MongoDB: {e}")
            import traceback
            traceback.print_exc()

    def start_streaming(self):
        # print("====[MONGODB] - Reading data from Kafka====")
        df = self.spark.readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", self.bootstrap_servers) \
            .option("subscribe", self.kafka_topic) \
            .option("startingOffsets", "latest") \
            .load()

        df_parsed = df.selectExpr("CAST(value AS STRING)") \
            .select(from_json(col("value"), self.schema).alias("data")) \
            .select("data.*")

        # print("====[MONGODB] - Starting write stream to MongoDB using foreachBatch====")
        query = df_parsed.writeStream \
            .foreachBatch(self.write_batch_to_mongo) \
            .outputMode("append") \
            .option("checkpointLocation", "./checkpoints/stream_to_mongo") \
            .start()

        query.awaitTermination()


if __name__ == "__main__":
    streamer = MongoStreamer()
    streamer.start_streaming()