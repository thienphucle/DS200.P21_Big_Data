# dashboard.py

import streamlit as st
from kafka import KafkaConsumer
import json
import pandas as pd
import time
from threading import Thread
from queue import Queue
from config import KAFKA_CONFIG, DASHBOARD_CONFIG

# Cấu hình Kafka
TOPIC = DASHBOARD_CONFIG['online_prediction_topic']
BOOTSTRAP_SERVERS = KAFKA_CONFIG['bootstrap_servers']

# Tạo hàng đợi an toàn luồng
message_queue = Queue()

# Hàm chạy Kafka Consumer trong luồng riêng
def consume_kafka_messages():
    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=BOOTSTRAP_SERVERS,
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        auto_offset_reset='latest',
        enable_auto_commit=True,
        group_id='streamlit_dashboard_group'
    )

    for message in consumer:
        message_queue.put(message.value)

# Chạy consumer thread
consumer_thread = Thread(target=consume_kafka_messages, daemon=True)
consumer_thread.start()

# Giao diện Streamlit
st.set_page_config(page_title="TikTok Prediction Dashboard", layout="wide")
st.title("📊 TikTok Video Interaction Prediction (Real-time)")
placeholder = st.empty()

# DataFrame để chứa lịch sử
history_df = pd.DataFrame(columns=["timestamp", "user_name", "vid_id", "prediction"])

# Vòng lặp cập nhật dashboard
while True:
    new_data = []
    while not message_queue.empty():
        msg = message_queue.get()
        new_data.append(msg)

    if new_data:
        new_df = pd.DataFrame(new_data)
        history_df = pd.concat([history_df, new_df], ignore_index=True)

        # Chuyển timestamp về datetime nếu cần
        history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])

        # Giữ lại 100 dòng mới nhất
        history_df = history_df.sort_values('timestamp', ascending=False).head(100)

        with placeholder.container():
            st.subheader("🔮 Latest Predictions")
            st.dataframe(history_df.sort_values("timestamp", ascending=False), use_container_width=True)

            st.line_chart(history_df.sort_values("timestamp")[["prediction"]])

    time.sleep(2)
