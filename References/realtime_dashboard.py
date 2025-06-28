# ----------- realtime_dashboard.py -----------
import streamlit as st
import pandas as pd
import time
import json
from confluent_kafka import Consumer, KafkaError
from datetime import datetime
from config import KAFKA_CONFIG

# Khởi tạo session state để lưu dữ liệu tích lũy
if 'accumulated_data' not in st.session_state:
    st.session_state.accumulated_data = pd.DataFrame(columns=["timestamp", "product_id", "content", "aspect_sentiment"])

st.set_page_config(page_title="Tiki Comment Analysis", layout="wide")

def create_kafka_consumer():
    conf = {
        'bootstrap.servers': KAFKA_CONFIG["bootstrap.servers"],
        'group.id': 'tiki-dashboard-group',
        'auto.offset.reset': 'latest',
        'enable.auto.commit': False
    }
    consumer = Consumer(conf)
    consumer.subscribe([KAFKA_CONFIG["output_topic"]])
    return consumer

def process_message(message):
    try:
        data = json.loads(message.value().decode('utf-8'))
        predictions = data.get('predictions', {})
        
        if isinstance(predictions, str):
            predictions = json.loads(predictions.replace("'", "\""))
        
        # Giữ lại tất cả aspect kể cả sentiment None
        return {
            "product_id": data.get("product_id", "N/A"),
            "content": data.get("processed_content", "N/A"),
            "predictions": predictions,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        }
    except Exception as e:
        st.error(f"Error processing message: {str(e)}")
        return None

def get_realtime_data(consumer):
    data = []
    start_time = time.time()
    
    while time.time() - start_time < 1:
        msg = consumer.poll(0.1)
        
        if msg is None:
            continue
            
        if msg.error() and msg.error().code() != KafkaError._PARTITION_EOF:
            continue
                
        processed = process_message(msg)
        if processed and processed['predictions']:
            data.append(processed)
    
    return data

st.title("📊 Tiki Comment Analysis - Live Feed")

# Khởi tạo consumer
consumer = create_kafka_consumer()
data_placeholder = st.empty()

try:
    while True:
        raw_data = get_realtime_data(consumer)
        
        if raw_data:
            # Tạo DataFrame mới
            new_data = []
            for record in raw_data:
                # Tạo chuỗi aspect:sentiment
                aspect_pairs = [
                    f"{aspect}: {sentiment}" 
                    for aspect, sentiment in record['predictions'].items()
                ]
                new_data.append({
                    "timestamp": record['timestamp'],
                    "product_id": record['product_id'],
                    "content": record['content'],
                    "aspect_sentiment": ", ".join(aspect_pairs)
                })
            
            new_df = pd.DataFrame(new_data)
            
            # Cập nhật dữ liệu tích lũy
            if not new_df.empty:
                st.session_state.accumulated_data = pd.concat(
                    [st.session_state.accumulated_data, new_df],
                    ignore_index=True
                ).tail(100)  # Giữ lại 100 bản ghi gần nhất
            
        # Hiển thị toàn bộ dữ liệu
        with data_placeholder.container():
            st.dataframe(
                st.session_state.accumulated_data,
                height=600,
                column_config={
                    "timestamp": "Timestamp",
                    "product_id": "Product ID",
                    "content": {"title": "Comment Text"},
                    "aspect_sentiment": {"title": "Aspect Sentiment Pairs"}
                },
                use_container_width=True
            )
        
        time.sleep(0.5)

finally:
    consumer.close()
    st.success("✅ Disconnected from Kafka")