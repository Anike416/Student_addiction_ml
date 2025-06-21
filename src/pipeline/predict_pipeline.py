import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = 'files/model.pkl'
            preprocessor_path = 'files/preprocessor.pkl'

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            data_scaled = preprocessor.transform(features)
            pred = model.predict(data_scaled)

            return pred
        
        except Exception as e:
            raise CustomException(e, sys)
        
class CustomData:
    def __init__(self,
                 age: int,
                 gender: str,
                 academic_level: str,
                 country: str,
                 avg_daily_usage_hours: float,
                 most_used_platform: str,
                 affects_academic_performance: str,
                 sleep_hours_per_night: float,
                 mental_health_score: int,
                 relationship_status: str,
                 conflicts_over_social_media: int
                 ):
        self.age = age
        self.gender = gender
        self.academic_level = academic_level
        self.country = country
        self.avg_daily_usage_hours = avg_daily_usage_hours
        self.most_used_platform = most_used_platform
        self.affects_academic_performance = affects_academic_performance
        self.sleep_hours_per_night = sleep_hours_per_night
        self.mental_health_score = mental_health_score
        self.relationship_status = relationship_status
        self.conflicts_over_social_media = conflicts_over_social_media

    def get_data_as_dataframe(self):
        try:
            data_dict = {
                'Age': [self.age],
                'Gender': [self.gender],
                'Academic_Level': [self.academic_level],
                'Country': [self.country],
                'Avg_Daily_Usage_Hours': [self.avg_daily_usage_hours],
                'Most_Used_Platform': [self.most_used_platform],
                'Affects_Academic_Performance': [self.affects_academic_performance],
                'Sleep_Hours_Per_Night': [self.sleep_hours_per_night],
                'Mental_Health_Score': [self.mental_health_score],
                'Relationship_Status': [self.relationship_status],
                'Conflicts_Over_Social_Media': [self.conflicts_over_social_media]
            }

            return pd.DataFrame(data_dict)
        
        except Exception as e:
            raise CustomException(e, sys)
        
        