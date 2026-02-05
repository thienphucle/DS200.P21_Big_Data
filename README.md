# ViTIED: Predicting TikTok KOL/KOC Engagement for Marketing Optimization

## Overview
This repository contains the **source code and dataset** accompanying the research paper:

> **Predicting Social Media Engagement of KOL/KOC for Marketing Optimization**  
> Ho Ngoc Mai, Duong Thi Hong Nhung, Le Ngoc Thien Phuc  
> Faculty of Information Science and Engineering,  
> University of Information Technology – VNU-HCM, Vietnam

This work proposes an **end-to-end system** for predicting TikTok influencer engagement and classifying interaction trends, integrating:
- automated data collection,
- real-time streaming analytics,
- feature engineering,
- machine learning and deep learning models.

The repository supports **reproducibility**, **extension**, and **further research** on influencer analytics and marketing optimization.

---

## Repository Structure
```text
DS200.P21_Big_Data/
├── Dataset/
│   ├── FE_Results/              # Processed dataset (35K+ videos)
|   ├── Preprocessed_Data/
|   ├── Raw_Data/
├── ModelResults/
├── Offline_System/
│   ├── __init__.py
│   ├── Bmodel.py
│   ├── Feature_Engineering.py
│   ├── Filtering.ipynb
│   ├── Preprocessing.py
│   ├── model.py
│   ├── vietnamese-stopwords-dash.txt
├── Online_System/
│   ├── checkpoints
│   ├── mongo_checkpoint
│   ├── enhanced_spark_checkpoint
│   ├── Dashboard.py
│   ├── Prediction.py
│   ├── Preprocessor.py
│   ├── Producer.py
│   ├── Stream_to_MongoDB.py
│   ├── config.py
│   ├── docker-compose.yml
│   ├── requirements.txt
│   ├── run_system.py
├── Results/
├── catboost_info/
├── src/
├── .gitattributes
├── .gitignore
└── README.md
