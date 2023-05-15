import os
import sys
import pickle

import numpy as np
import pandas as pd

from src.exception import CustomException

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
