# Dashboard.py

import streamlit as st
from kafka import KafkaConsumer
import json
import pandas as pd

from config import KAFKA_CONFIG

# Thiết lập Kafka Consumer
consumer = KafkaConsumer(
    'video_predictions',  # topic để lắng nghe kết quả dự đoán
    bootstrap_servers=KAFKA_CONFIG['bootstrap_servers'],
    value_deserializer=lambda m: json.loads(m.decode('utf-8')),
    auto_offset_reset='latest',
    enable_auto_commit=True,
    group_id='dashboard_group'
)

# Cấu hình giao diện Streamlit
st.set_page_config(page_title="TikTok Prediction Dashboard", layout="wide")
st.title("📊 TikTok Video Growth Prediction")
st.markdown("Real-time predictions streamed from Kafka")

# Khởi tạo DataFrame trống để lưu kết quả
prediction_data = []

# Stream dữ liệu từ Kafka và cập nhật giao diện
placeholder = st.empty()

while True:
    for message in consumer:
        try:
            result = message.value  # dict chứa kết quả
            prediction_data.append(result)

            # Hiển thị bảng
            df = pd.DataFrame(prediction_data).sort_values("timestamp", ascending=False)
            placeholder.dataframe(df.head(20), use_container_width=True)

        except Exception as e:
            st.error(f"Lỗi đọc dữ liệu Kafka: {e}")
