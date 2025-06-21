from flask import Flask, request, render_template
import pandas as pd
from src.pipeline.predict_pipeline import CustomData,PredictPipeline
from src.utils import load_object

app = Flask(__name__)

# Load model and preprocessor
model = load_object('files/model.pkl')
preprocessor = load_object('files/preprocessor.pkl')

@app.route('/')
def welcome():
    return render_template('home.html')

@app.route('/predict-form')
def form():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = CustomData(
            age=int(request.form['age']),
            gender= request.form['gender'],
            academic_level=request.form['academic_level'],
            country=request.form['country'],
            avg_daily_usage_hours=float(request.form['usage']),
            most_used_platform= request.form['platform'],
            affects_academic_performance=request.form['affects'],
            sleep_hours_per_night=float(request.form['sleep']),
            mental_health_score= int(request.form['mental']),
            relationship_status= request.form['relationship'],
            conflicts_over_social_media= int(request.form['conflicts'])
        )

        pred_df=data.get_data_as_dataframe()
        print(pred_df)
        
        pipeline=PredictPipeline()
        
        prediction=pipeline.predict(pred_df)
        
        if prediction <= 3:
            concern = "✅ Low (Healthy usage)"
        elif prediction <= 6:
            concern = "⚠️ Moderate (Monitor usage)"
        elif prediction <= 8:
            concern = "🚨 High (Consider reducing)"
        else:
            concern = "🔥 Critical (Likely addicted)"
        return render_template('index.html',prediction=prediction[0],concern=concern)


        

    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)
