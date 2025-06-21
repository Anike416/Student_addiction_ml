import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransConfig:
    preprocessor_obj_file_path: str = os.path.join('files', 'preprocessor.pkl')

class DataTrans:
    def __init__(self):
        self.data_trans_config = DataTransConfig()

    def get_data(self):
        try:
            num_features = ['Age', 'Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night', 'Mental_Health_Score', 'Conflicts_Over_Social_Media']
            cat_features = ['Gender', 'Academic_Level', 'Country', 'Most_Used_Platform', 'Affects_Academic_Performance', 'Relationship_Status']

            num_pipeline = Pipeline(steps=[
                
                ("scaler", StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ("scaler", StandardScaler(with_mean=False))
                
            ])

            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, num_features),
                ("cat_pipeline", cat_pipeline, cat_features)
            ])

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_trans(self, train_path, test_path):
        try:
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            preprocessing_obj = self.get_data()

            target_col = 'Addicted_Score'

            input_feature_train_df = train_data.drop(columns=[target_col])
            target_feature_train_df = train_data[target_col]

            input_feature_test_df = test_data.drop(columns=[target_col])
            target_feature_test_df = test_data[target_col]

            logging.info("Applying preprocessing on train and test data")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            input_feature_train_arr_dense = input_feature_train_arr.toarray()
            input_feature_test_arr_dense = input_feature_test_arr.toarray()

# Now safely concatenate
            train_arr = np.c_[
                input_feature_train_arr_dense,
                target_feature_train_df.to_numpy().reshape(-1, 1)
            ]
            
            test_arr = np.c_[
                input_feature_test_arr_dense,
                target_feature_test_df.to_numpy().reshape(-1, 1)
            ]
            
            save_object(
                file_path=self.data_trans_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info("Saved preprocessor object")
            
            return train_arr, test_arr, self.data_trans_config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)
