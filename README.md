# 📱 Social Media Addiction Score Predictor
This is a web application that predicts a student's social media addiction score (1–10) based on their behavioral, lifestyle, and psychological factors.
It also provides a concern level assessment to guide users about their digital wellbeing.

| Version        | Status     |                           Link                             |
|----------------|------------|------------------------------------------------------------|
| Streamlit App  | ✅ Local   | Run with Streamlit.py            http://192.168.1.4:8502
| Flask App      | ✅ Local   | Run with app.py            http://127.0.0.1:5000

## 🌟 Key Features
🔢 Predicts a numerical addiction score (1 to 10)

🧠 Provides a concern level (Low, Moderate, High, Critical)

📋 Collects input about screen time, sleep, mental health, academics, etc.

🧮 Built using a trained machine learning model (Random Forest, 98% R²)

🌐 Clean UI built with Flask + HTML/CSS

✅ Modular code structure using OOP (CustomData, PredictPipeline)

🎯 Easy to extend with new features or deploy online


## 🖥️ Tech Stack
Layer	Tools Used
💡 Model	Scikit-learn (Random Forest Regressor)

🔄 Preprocess	OneHotEncoding, StandardScaler, Pipeline

🌐 Frontend	HTML5, CSS3

🧠 Backend	Flask (Python)

📦 Deployment	(Localhost / ready for Render/Heroku)


## 📊 Dataset Overview
Source: Simulated / collected student social media usage data

Size: 705 entries × 13 features

Features include:

Age, Gender, Academic Level, Avg_Daily_Usage_Hours
Sleep Hours, Mental Health Score, Conflicts Over Social Media
Platform, Country, Relationship Status, etc.



## 🧠 ML Pipeline Highlights 

📌 Target: Addicted_Score (range 1–10)

🔍 Feature Engineering: Categorical encoding, scaling, missing value handling

📈 Model used: RandomForestRegressor

🧪 Evaluation: Achieved R² = 0.98 on validation set

🔗 ML pipeline and model are saved using pickle
