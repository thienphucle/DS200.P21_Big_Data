import streamlit as st
import pandas as pd
import pymongo


class Dashboard:
    def __init__(self, mongo_uri="mongodb://localhost:27017", db_name="TikTok_Database", collection="predictions"):
        self.client = pymongo.MongoClient(mongo_uri)
        self.collection = self.client[db_name][collection]
        self.history_df = pd.DataFrame()

    def load_data(self, limit=100):
        """Đọc dữ liệu mới nhất từ MongoDB"""
        try:
            cursor = list(self.collection.find().sort("timestamp", -1).limit(limit))
            if cursor:
                df = pd.DataFrame(cursor)
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df[[
                    "user_name", "vid_id", "timestamp",
                    "prediction_view", "prediction_like", "prediction_comment",
                    "prediction_share", "prediction_save",
                    "growth_category", "confidence"
                ]]

                # print(f"====[DASHBOARD] - Dữ liệu đọc từ MongoDB: \n{df}")

                # Ghép với dữ liệu cũ, tránh trùng lặp
                self.history_df = pd.concat([df, self.history_df]) \
                    .drop_duplicates(subset=["vid_id", "timestamp"]) \
                    .sort_values("timestamp", ascending=False) \
                    .head(100)
        except Exception as e:
            st.warning(f"⚠ MongoDB read error: {e}")

    def run(self):
        st.set_page_config(layout="wide")
        st.title("📊 Predicting Social Media Engagement of KOL/KOC for Marketing Optimization")

        # Thông báo trạng thái kết nối Mongo
        try:
            self.client.server_info()
            st.success("✅ Connected to MongoDB successfully.")
        except Exception as e:
            st.error(f"❌ Failed to connect to MongoDB: {e}")
            return

        # Nút làm mới dữ liệu
        if st.button("🔄 Refresh Predictions"):
            self.load_data()

        if not self.history_df.empty:
            st.subheader("🔮 Latest Predictions")
            
            st.dataframe(self.history_df, use_container_width=True)

            # Hiển thị biểu đồ
            chart_df = self.history_df.sort_values("timestamp")
            st.subheader("📈 Interaction Growth Trend (View / Like / Comment / Share / Save)")

            st.line_chart(chart_df[[
                "prediction_view",
                "prediction_like",
                "prediction_comment",
                "prediction_share",
                "prediction_save"
            ]])
        else:
            st.info("ℹ️ No data available yet.")


if __name__ == "__main__":
    processor = Dashboard()
    # print("===============STARTING READING DATA & PRINTING DATA ON DASHBOARD===============")
    processor.run()
