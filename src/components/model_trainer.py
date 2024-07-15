import os
import sys
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            target_col="Churn"
            columns = ['Age', 'Gender_Male',"Gender_Female", 'Tenure', 'Usage Frequency',
       'Support Calls_Basic','Support Calls_Premium', 'Payment Delay', 'Subscription Type',
       'Contract Length_Annual','Contract Length_Monthly','Contract Length_Quarterly', 'Last Interaction', 'Churn']
           
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            aa=test_array[:,-1]
            unique_values = np.unique(aa)
            # print(f"Uniques val:{unique_values}")
            models = {
                "Gradient Boosting": GradientBoostingClassifier(),
                # "Linear Regression": LinearRegression(),
                # "AdaBoost Boosting": AdaBoostClassifier(),
            }
            

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models)
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            acc_score = accuracy_score(y_test, predicted)
            return acc_score
            



            
        except Exception as e:
            raise CustomException(e,sys) 