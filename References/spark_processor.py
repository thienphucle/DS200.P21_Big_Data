# ----------- spark_processor.py -----------
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, udf, struct, to_json
from pyspark.sql.types import *
import torch
import logging
import os
import json
import pickle
import traceback
from transformers import BertTokenizerFast, BertModel, BertConfig
from config import KAFKA_CONFIG, SPARK_CONFIG, MODEL_CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("SparkProcessor")

# Global resources
_resources_initialized = False
_tokenizer = None
_model = None
_label_mapping = None
_aspect_list = ['BATTERY', 'CAMERA', 'DESIGN', 'FEATURES', 'GENERAL',
               'PERFORMANCE', 'PRICE', 'SCREEN', 'SER&ACC', 'STORAGE']

class BertForMultiLabelClassification(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Load config với vocab_size chính xác
        config = BertConfig.from_pretrained(
            MODEL_CONFIG["model_path"],
            vocab_size=119547
        )
        self.bert = BertModel.from_pretrained(
            MODEL_CONFIG["model_path"],
            config=config,
            ignore_mismatched_sizes=True
        )
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, 40)  # 10 aspects * 4 classes
        self.dropout = torch.nn.Dropout(0.3)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = self.dropout(outputs.pooler_output)
        logits = self.classifier(pooled_output)
        return logits.view(-1, 10, 4)

def _initialize_resources():
    global _resources_initialized, _tokenizer, _model, _label_mapping
    
    if not _resources_initialized:
        try:
            logger.info("🔄 Initializing resources...")
            
            # Load tokenizer
            _tokenizer = BertTokenizerFast.from_pretrained(
                MODEL_CONFIG["model_path"],
                do_lower_case=False
            )
            
            # Khởi tạo model
            _model = BertForMultiLabelClassification()
            
            # Load model weights
            model_path = os.path.join(MODEL_CONFIG["model_path"], "pytorch_model.bin")
            _model.load_state_dict(
                torch.load(model_path, map_location=torch.device('cpu')),
                strict=False
            )
            _model.eval()
            
            # Load và validate label mapping
            with open(MODEL_CONFIG["label_mapping_path"], 'rb') as f:
                raw_mapping = pickle.load(f)
                
            # Chuẩn hóa label mapping
            _label_mapping = {
                0: raw_mapping.get(0, "None"),
                1: raw_mapping.get(1, "Positive"),
                2: raw_mapping.get(2, "Neutral"),
                3: raw_mapping.get(3, "Negative")
            }
            
            # Validate mapping
            expected_labels = ["None", "Positive", "Neutral", "Negative"]
            if list(_label_mapping.values()) != expected_labels:
                raise ValueError(f"Invalid label mapping. Expected {expected_labels} but got {list(_label_mapping.values())}")
            
            _resources_initialized = True
            logger.info("✅ Resources initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Initialization failed: {str(e)}\n{traceback.format_exc()}")
            raise

def _predict_udf(processed_text):
    try:
        if not processed_text or len(processed_text.strip()) == 0:
            return json.dumps({"error": "Empty input"})
            
        if not _resources_initialized:
            _initialize_resources()
        
        # Tokenization
        inputs = _tokenizer(
            processed_text,
            max_length=256,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        
        # Inference
        with torch.no_grad():
            logits = _model(inputs['input_ids'], inputs['attention_mask'])
        
        # Post-processing
        preds = torch.argmax(logits, dim=-1).squeeze().numpy()
        predictions = {
            aspect: _label_mapping[preds[i]] 
            for i, aspect in enumerate(_aspect_list) 
            if i < len(preds)
        }
        
        return json.dumps(predictions, ensure_ascii=False)

    except Exception as e:
        logger.error(f"🔴 Prediction error: {str(e)}\n{traceback.format_exc()}")
        return json.dumps({"error": str(e)})

class SparkStreamProcessor:
    def __init__(self):
        self.spark = self._initialize_spark()
        logger.info("🚀 Spark session initialized")
        
    def _initialize_spark(self):
        return SparkSession.builder \
            .appName("TikiAspectAnalysis") \
            .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0") \
            .config("spark.executor.memory", SPARK_CONFIG["executor_memory"]) \
            .config("spark.driver.memory", SPARK_CONFIG["driver_memory"]) \
            .config("spark.sql.shuffle.partitions", "4") \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
            .getOrCreate()

    def process_stream(self):
        input_schema = StructType([
            StructField("product_id", StringType()),
            StructField("processed_content", StringType())
        ])

        predict_udf_func = udf(_predict_udf, StringType())

        try:
            # Đọc stream từ Kafka
            stream_df = self.spark.readStream \
                .format("kafka") \
                .option("kafka.bootstrap.servers", KAFKA_CONFIG["bootstrap.servers"]) \
                .option("subscribe", KAFKA_CONFIG["input_topic"]) \
                .option("startingOffsets", "latest") \
                .option("failOnDataLoss", "false") \
                .load()
            
            # Xử lý dữ liệu
            processed_df = stream_df \
                .select(from_json(col("value").cast("string"), input_schema).alias("data")) \
                .select("data.*") \
                .withColumn("predictions", predict_udf_func(col("processed_content"))) \
                .select(
                    col("product_id").alias("key"),
                    to_json(struct("product_id", "processed_content", "predictions")).alias("value")
                )
            
            # Ghi kết quả
            query = processed_df \
                .writeStream \
                .format("kafka") \
                .option("kafka.bootstrap.servers", KAFKA_CONFIG["bootstrap.servers"]) \
                .option("topic", KAFKA_CONFIG["output_topic"]) \
                .option("checkpointLocation", SPARK_CONFIG["checkpoint_location"]) \
                .outputMode("append") \
                .start()
            
            logger.info(f"📡 Started streaming to {KAFKA_CONFIG['output_topic']}")
            return query
            
        except Exception as e:
            logger.error(f"🔥 Stream processing failed: {str(e)}\n{traceback.format_exc()}")
            raise

if __name__ == "__main__":
    processor = SparkStreamProcessor()
    try:
        query = processor.process_stream()
        query.awaitTermination()
    except KeyboardInterrupt:
        logger.info("🛑 Shutting down gracefully...")
        query.stop()
        query.awaitTermination()
    except Exception as e:
        logger.error(f"💀 Critical failure: {str(e)}\n{traceback.format_exc()}")