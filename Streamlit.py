import streamlit as st
import pickle
import pandas as pd

# Load model and preprocessor
with open('files/model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('files/preprocessor.pkl', 'rb') as file:
    preprocessor = pickle.load(file)

def main():
    st.title('📱 Social Media Addiction Score Predictor')

    st.markdown("""
    This app predicts how addicted a student is to social media 
    (on a scale of 1–10) based on their lifestyle, sleep, mental health, and usage habits.
    """)

    # User Inputs
    age = st.number_input("Age", min_value=10, max_value=100, step=1)
    gender = st.selectbox("Gender", ["Male", "Female"])
    academic_level = st.selectbox("Academic Level", ["High School", "Undergraduate", "Graduate"])
    country = st.text_input("Country", "India")
    avg_daily_usage = st.number_input("Avg Daily Social Media Usage (Hours)", min_value=0.0, max_value=24.0, step=0.1)
    platform = st.selectbox("Most Used Platform", ["Instagram", "YouTube", "WhatsApp", "Facebook", "Twitter", ""])
    affects_academics = st.selectbox("Does it affect academic performance?", ["Yes", "No"])
    sleep_hours = st.number_input("Sleep Hours per Night", min_value=0.0, max_value=24.0, step=0.1)
    mental_health_score = st.slider("Mental Health Score (1=Poor, 10=Excellent)", 1, 10)
    relationship_status = st.selectbox("Relationship Status", ["Single", "In a Relationship", "Complicated", "Prefer not to say"])
    conflicts = st.slider("Conflicts Over Social Media (times per week)", 0, 10)
    


    # Prediction logic
    if st.button("Predict Addiction Score"):
        input_df = pd.DataFrame({
            "Age": [age],
            "Gender": [gender],
            "Academic_Level": [academic_level],
            "Country": [country],
            "Avg_Daily_Usage_Hours": [avg_daily_usage],
            "Most_Used_Platform": [platform],
            "Affects_Academic_Performance": [affects_academics],
            "Sleep_Hours_Per_Night": [sleep_hours],
            "Mental_Health_Score": [mental_health_score],
            "Relationship_Status": [relationship_status],
            "Conflicts_Over_Social_Media": [conflicts]
        })
       
        processed_input = preprocessor.transform(input_df)
        prediction = model.predict(processed_input)[0]

        st.success(f"📊 Predicted Social Media Addiction Score: **{prediction} / 10**")
        concern_level = ""
        emoji = ""

        if prediction <= 3:
         concern_level = "✅ Low"
         emoji = "😊 Healthy usage"
        elif prediction <= 6:
         concern_level = "⚠️ Moderate"
         emoji = "😐 Monitor your habits"
        elif prediction <= 8:
         concern_level = "🚨 High"
         emoji = "😟 Consider reducing usage"
        else:
         concern_level = "🔥 Critical"
         emoji = "😨 Strong signs of addiction"
     
        st.subheader("🧠 Concern Level Based on Score")
        st.info(f"**{concern_level}** — {emoji}")
if __name__ == '__main__':
    main()