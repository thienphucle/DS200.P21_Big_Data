from multiprocessing import Process
import subprocess
import time

def run_producer():
    subprocess.run(["python", "Producer.py"])

def run_prediction():
    subprocess.run(["python", "Prediction.py"])

def run_mongo_streamer():
    subprocess.run(["python", "Stream_to_MongoDB.py" ])

def run_dashboard():
    subprocess.run(["streamlit", "run", "Dashboard.py"])


if __name__ == "__main__":
    print("🔁 Starting TikTok Online Prediction System...")

    p1 = Process(target=run_producer)
    p2 = Process(target=run_prediction)
    p3 = Process(target=run_mongo_streamer)
    p4 = Process(target=run_dashboard)

    p1.start()
    time.sleep(1)  # Cho Producer chạy trước
    p2.start()
    time.sleep(1)
    p3.start()
    time.sleep(1)
    p4.start()

    p1.join()
    p2.join()
    p3.join()
    p4.join()
