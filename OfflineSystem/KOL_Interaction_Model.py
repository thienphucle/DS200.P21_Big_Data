import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

from xgboost import XGBRegressor

from sklearn.cluster import KMeans
import joblib

class AutoModelSelector:
    def __init__(self, models: dict):
        self.models = models
        self.best_model = None
        self.best_name = None
        self.results = {}

    def train_and_select(self, X_train, X_test, y_train, y_test):
        for name, model in self.models.items():
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("model", model)
            ])
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            self.results[name] = mse
            print(f"[{name}] MSE = {mse:.4f}")

            if self.best_model is None or mse < self.results[self.best_name]:
                self.best_model = pipe
                self.best_name = name

        print(f"\n✅ Best Model: {self.best_name} with MSE = {self.results[self.best_name]:.4f}")
        return self.best_model, self.best_name, self.results


class KOLInteractionPipeline:
    def __init__(self, video_feat_path, user_feat_path):
        self.video_feat_path = video_feat_path
        self.user_feat_path = user_feat_path
        self.video_model = None
        self.kmeans_model = None

    def load_data(self):
        df_video = pd.read_csv(self.video_feat_path)
        df_user = pd.read_csv(self.user_feat_path)
        return df_video, df_user

    def train_video_model(self, df_video):
        X = df_video.drop(columns=["future_view_growth_per_hour", "user_name", "vid_id"], errors="ignore")
        y = df_video["future_view_growth_per_hour"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Danh sách các mô hình để thử
        models = {
            "LinearRegression": LinearRegression(),
            "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
            "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
            "SVR": SVR(),
            "XGBoost": XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
        }

        selector = AutoModelSelector(models)
        best_model, best_name, all_results = selector.train_and_select(X_train, X_test, y_train, y_test)

        # Lưu model tốt nhất
        self.video_model = best_model
        joblib.dump(best_model, f"best_video_model_{best_name}.pkl")

        return all_results

    def cluster_users(self, df_user, n_clusters=3):
        scaler = StandardScaler()
        X_user = df_user.drop(columns=["user_name"], errors="ignore")
        X_scaled = scaler.fit_transform(X_user)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)

        df_user["cluster"] = cluster_labels

        df_user["recommend_booking"] = df_user.apply(
            lambda row: "Yes" if row["avg_engagement_growth"] > 500 or row["cluster"] == 2 else "No",
            axis=1
        )

        self.kmeans_model = kmeans
        joblib.dump(kmeans, "user_kmeans_model.pkl")

        print("[TASK 2] Clustering completed.")
        return df_user

    def run(self):
        df_video, df_user = self.load_data()

        print("🔧 Training Task 1: Auto Model Selection for video interaction prediction...")
        model_scores = self.train_video_model(df_video)

        print("\n🔍 Running Task 2: Clustering user trend features and booking suggestion...")
        df_user_clustered = self.cluster_users(df_user)

        df_user_clustered.to_csv("user_clustering_results.csv", index=False)
        print("✅ Output saved: user_clustering_results.csv")


# ---------- RUN PIPELINE ---------- #
if __name__ == "__main__":
    pipeline = KOLInteractionPipeline(
        video_feat_path=r"D:\UIT\DS200\your_folder\video_features.csv",
        user_feat_path=r"D:\UIT\DS200\your_folder\user_trend_features.csv"
    )
    pipeline.run()
