import streamlit as st
import pandas as pd
import pymongo
import time


class DashboardMongoApp:
    def __init__(self, mongo_uri="mongodb://localhost:27017", db_name="TikTokPrediction", collection="predictions"):
        self.client = pymongo.MongoClient(mongo_uri)
        self.collection = self.client[db_name][collection]
        self.history_df = pd.DataFrame()

    def run(self):
        st.set_page_config(layout="wide")
        st.title("📊 TikTok Video Predictions (MongoDB)")
        placeholder = st.empty()

        while True:
            try:
                cursor = list(self.collection.find().sort("timestamp", -1).limit(100))
                if cursor:
                    df = pd.DataFrame(cursor)
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    df = df[["timestamp", "user_name", "vid_id", "prediction"]]

                    self.history_df = pd.concat([df, self.history_df]).drop_duplicates().head(100)

                    with placeholder.container():
                        st.subheader("🔮 Latest Predictions")
                        st.dataframe(self.history_df, use_container_width=True)
                        st.line_chart(self.history_df.sort_values("timestamp")[["prediction"]])
            except Exception as e:
                st.warning(f"⚠ MongoDB read error: {e}")

            time.sleep(3)
