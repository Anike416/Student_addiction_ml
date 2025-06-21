import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import (GradientBoostingRegressor,
                              AdaBoostRegressor,
                              RandomForestRegressor
                              )
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge,Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_model

@dataclass
class ModelTrainingConfig:
    trained_model_file_path=os.path.join('files',"model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainingConfig()
        
        
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test data input")
            X_train,X_test,y_train,y_test=(
                train_array[:,:-1],
                test_array[:,:-1],
                train_array[:,-1],
                test_array[:,-1]
            )
            models=[
             ('Linear Regression',LinearRegression()),
             ('Lasso',Lasso()),
             ('Ridge',Ridge()),
             ('KNeighbor',KNeighborsRegressor()),
             ('DecisionTree',DecisionTreeRegressor()),
             ('RandomForest',RandomForestRegressor()) ,
             ('GradientBooster' , GradientBoostingRegressor()),
             ('Adaboost',AdaBoostRegressor()),
             ('XGBoost',XGBRegressor()),

            ]
            
            param_grid = {
                'GradientBooster':{
                    'learning_rate':[0.01,0.1,1],
                    'n_estimators':[50,100,200]
                },
                'DecisionTree' :{
                    'max_depth':[3,5,10],
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
                    
                },
                'KNeighbor':{
                    'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute'],
                    'n_neighbors':[5,7,10]                    
                    
                },
                'RandomForest': {
                                 'n_estimators': [10, 50, 100, 200],
                                 'max_depth': [3, 5, 10, 20]
                                 },
                'Adaboost': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 1.0]
                },
                'XGBoost': {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 5, 7],
                   'learning_rate': [0.01, 0.1, 0.2]
                },
                'Linear Regression': {
                    # No hyperparameters to tune for basic LinearRegression
                },
                'Lasso': {
                    'alpha': [0.01, 0.1, 1.0, 10]
                },
                'Ridge': {
                    'alpha': [0.01, 0.1, 1.0, 10]
                }
            }
            
            # Dictionary to store best results
            best_models = {}
            
            # Perform GridSearchCV
            for name, model in models:
                if name in param_grid and param_grid[name]:
                    grid = GridSearchCV(model, param_grid[name], cv=5, scoring='r2', n_jobs=-1)
                    grid.fit(X_train, y_train)
                    best_models[name] = {
                        'best_score': grid.best_score_,
                                    'best_params': grid.best_params_,
                        'best_estimator': grid.best_estimator_
                    }
                else:
                    # Fit directly for models with no hyperparameters
                    model.fit(X_train, y_train)
                    score = model.score(X_test, y_test)
                    best_models[name] = {
                        'best_score': score,
                        'best_params': 'Default',
                        'best_estimator': model
                    }
                    
            best_model_name = max(best_models, key=lambda x: best_models[x]['best_score'])
            best_model = best_models[best_model_name]['best_estimator']
            
            
            
            logging.info("Best found model on both trainig and testing dataset")
            
            save_object(file_path=self.model_trainer_config.trained_model_file_path,
                        obj=best_model)
            
            predicted=best_model.predict(X_test)
            
            r2=r2_score(y_test,predicted)
 
            return best_model_name,r2
        
        
        except Exception as e:
            raise CustomException(e,sys)   