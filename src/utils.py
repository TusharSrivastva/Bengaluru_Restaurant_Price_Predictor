import os
import sys
import pickle

import numpy as np
import pandas as pd

from src.exception import CustomException
from sklearn.metrics import r2_score

# Function for pickling objects
def save_object(file_path, obj) -> None:
    """
    This function takes file path and object as input and saves the object
    at specified path.

    * file_path: Path of the location where object is to be stored
    * obj: Object to be pickled
    * dir_path: Variable for creating directory

    """
    try:
        # Extracting directory from file_path
        dir_path: str = os.path.dirname(file_path)

        # Creating directory
        os.makedirs(dir_path, exist_ok=True)

        # Pickling object
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


# Function for loading pickled objects
def load_object(file_path) -> pickle:
    """
    This function is used to load pickled objects.

    """
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)

# Function to train and evaluate models
def evaluate_models(X_train: pd.DataFrame,
                    y_train: pd.Series,
                    X_test: pd.DataFrame,
                    y_test: pd.Series,
                    models: dict) -> dict:
    """
    This function trains models and evaluates their r2 score
        * report: Dictionary containing model names and r2 scores
        * model: Store model from the dictionary of models
        * y_test_pred: Predicted target variable from X_test data
        * test_model_score: r2 score of the model
    """
    try:
        report: dict = {}

        for i in range(len(list(models))):
            # Get model
            model = list(models.values())[i]

            # Train model
            model.fit(X_train,y_train)

            # Predict
            y_test_pred: pd.Series = model.predict(X_test)

            # Evaluate r2 score
            test_model_score: float = r2_score(y_test, y_test_pred)

            # Add r2 score to dictionary
            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
