import sys
import os

import numpy as np
import pandas as pd

from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import (
    LinearRegression, 
    Ridge,
    Lasso
)
from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor
)

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import evaluate_models

from dataclasses import dataclass

from typing import Type, Tuple


@dataclass
class ModelTrainerConfig:
    """
    class ModelTrainerCofig is used to initialize the path to pickled model file

    """
    trained_model_path: str = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    """
    class ModelTrainer:
        * __init__() -> None
        * initiate_model_training(self, 
                                X_train: pd.DataFrame, 
                                y_train: pd.Series, 
                                X_test: pd.DataFrame, 
                                y_test: pd.Series) -> Tuple[float,
                                                            str]:
        
        This class is used to implement model training. Various models are
        imported and trained on the X_train and y_train datasets.

    """
    def __init__(self) -> None:
        self.model_trainer_config: Type[ModelTrainerConfig] = ModelTrainerConfig()
    

    def initiate_model_training(self, 
                                X_train: pd.DataFrame, 
                                y_train: pd.Series, 
                                X_test: pd.DataFrame, 
                                y_test: pd.Series) -> Tuple[float,
                                                            str]:
        """
        This function is used to implement model training. First the feature dataset and 
        label series are imported. Further they are converted to numpy arrays and the
        models are trained on them. The function returns the r2 score of the best model
        and the path to the pickled model object.
        * models: Dictionary containing model names and models
        * model_report: Dictionary containing model r2 score and model name
        * best_model_score: r2 score of the best model
        * best_model_name: Name of the best model

        """
        try:
            logging.info("Model training started")

            logging.info("Converting X and y to numpy arrays")

            # Converting X and y to numpy arrays
            X_train.to_numpy()
            X_test.to_numpy()
            y_train.to_numpy()
            y_test.to_numpy()

            logging.info("X and y converted to numpy arrays")

            logging.info("Creating a dictionary of training models")
            
            # Dictionary of training models 
            models: dict = {
                "SVR": SVR(),
                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "XGBRegressor": XGBRegressor(), 
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
                }
            
            logging.info("Dictionary of training models created")

            logging.info("Model training started")

            # Training models
            model_report: dict = evaluate_models(X_train=X_train,
                                                 y_train=y_train,
                                                 X_test=X_test,
                                                 y_test=y_test,
                                                 models=models)
            
            logging.info("Model training complete")

            logging.info("Getting the best model info")

            # Best model score
            best_model_score: float = max(sorted(model_report.values()))

            # Best model name
            best_model_name: str = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            # Getting the best model from models dictionary
            best_model = models[best_model_name]

            print("Best model is:", best_model_name)
            print("r2 score is", best_model_score)

            if best_model_score<0.6:
                raise CustomException("No best model found")
            
            logging.info("Have the best model info")

            logging.info("Saving the best model")

            # Saving model as pickled file
            save_object(
                file_path=self.model_trainer_config.trained_model_path,
                obj=best_model
            )

            logging.info("Best model saved")

            logging.info("Model training ended")

            return (
                best_model_score,
                self.model_trainer_config.trained_model_path
                )


        except Exception as e:
            raise CustomException(e, sys)

