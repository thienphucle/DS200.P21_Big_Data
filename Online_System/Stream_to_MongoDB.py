from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StringType, DoubleType
from config import KAFKA_CONFIG, DASHBOARD_CONFIG, MONGODB_CONFIG
from datetime import datetime



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
            'org.mongodb.spark:mongo-spark-connector_2.12:10.1.1'
        ]

        self.spark = SparkSession.builder.master("local[*]")\
            .appName("KafkaToMongo")\
            .config("spark.jars.packages", ",".join(packages))\
            .config("spark.mongodb.write.connection.uri", self.mongo_uri) \
            .getOrCreate()



        self.schema = StructType() \
            .add("user_name", StringType()) \
            .add("vid_id", StringType()) \
            .add("timestamp", StringType()) \
            .add("prediction", DoubleType()) \
            .add("confidence", DoubleType()) \
            
        
    def start_streaming(self):
        print("====[MONGODB] - Reading data from Output Kafka Topic====")
        df = self.spark.readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", self.bootstrap_servers) \
            .option("subscribe", self.kafka_topic) \
            .option("startingOffsets", "latest") \
            .load()

        print("====[MONGODB] - Parsing output data====")
        df_parsed = df.selectExpr("CAST(value AS STRING)") \
            .select(from_json(col("value"), self.schema).alias("data")) \
            .select("data.*")

        print("====[MONGODB] - Writing output data to MongoDB====")
        query = df_parsed.writeStream \
            .format("mongodb") \
            .option("database", self.mongo_db) \
            .option("collection", self.mongo_collection) \
            .option("checkpointLocation", "./mongo_checkpoint/") \
            .outputMode("append") \
            .start()

        print(f"✅ Spark Mongo Stream started from Kafka topic '{self.kafka_topic}' → MongoDB")
        query.awaitTermination()
    
if __name__ == "__main__":

    processor = MongoStreamer()
    print("STARTING READING OUTPUT & STREAMING DATA TO MONOGODB")
    processor.start_streaming()
