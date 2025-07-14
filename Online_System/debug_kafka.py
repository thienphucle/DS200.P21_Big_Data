from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("KafkaTopicViewer") \
    .master("local[*]") \
    .config("spark.jars.packages", 
        "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.5,org.apache.kafka:kafka-clients:3.6.0") \
    .getOrCreate()

df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "online_prediction_topic") \
    .option("startingOffsets", "latest") \
    .load()

df.selectExpr("CAST(key AS STRING)", "CAST(value AS STRING)") \
    .writeStream \
    .format("console") \
    .option("truncate", False) \
    .start() \
    .awaitTermination()