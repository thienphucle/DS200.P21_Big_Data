from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StringType, DoubleType
from config import KAFKA_CONFIG, DASHBOARD_CONFIG, MONGODB_CONFIG


class MongoStreamer:
    def __init__(self):
        self.kafka_topic = DASHBOARD_CONFIG['online_prediction_topic']
        self.bootstrap_servers = KAFKA_CONFIG['bootstrap_servers']
        self.mongo_uri = MONGODB_CONFIG['uri']
        self.mongo_db = MONGODB_CONFIG['database']
        self.mongo_collection = MONGODB_CONFIG['collection']

        self.spark = SparkSession.builder \
            .appName("KafkaToMongo") \
            .config("spark.mongodb.write.connection.uri", self.mongo_uri) \
            .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:10.1.1") \
            .getOrCreate()

        self.schema = StructType() \
            .add("vid_id", StringType()) \
            .add("user_name", StringType()) \
            .add("prediction", DoubleType()) \
            .add("timestamp", StringType())

    def start_streaming(self):
        df = self.spark.readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", self.bootstrap_servers) \
            .option("subscribe", self.kafka_topic) \
            .option("startingOffsets", "latest") \
            .load()

        df_parsed = df.selectExpr("CAST(value AS STRING)") \
            .select(from_json(col("value"), self.schema).alias("data")) \
            .select("data.*")

        query = df_parsed.writeStream \
            .format("mongodb") \
            .option("database", self.mongo_db) \
            .option("collection", self.mongo_collection) \
            .option("checkpointLocation", "./mongo_checkpoint/") \
            .outputMode("update") \
            .start()

        print(f"✅ Spark Mongo Stream started from Kafka topic '{self.kafka_topic}' → MongoDB")
        query.awaitTermination()
