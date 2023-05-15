import pandas as pd
import numpy as np
import sys

from src.exception import CustomException
from src.logger import logging

class Transformation_functions:
    """
    class Transfomation_functions:
        * __init__() -> None
        * preprocess(df: pd.DataFrame) -> pd.DataFrame
        * prep_rate(df: pd.DataFrame) -> pd.DataFrame
        * X_y_split(df: pd.DataFrame) -> pd.DataFrame

        This class contains necessary functions for data transformation.
    """

    def __init__(self) -> None:
        pass
    
    # Function for important preprocessing
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """ 
        This function drops rows with NULL values in columns
        'approx_cost(for two people)' and 'rest_type'. Then
        the dataset is exploded by 'rest_type.
  
        """
        try:
            logging.info("preproces() has begun")

            # Dropping rows where price or rest_type in NULL
            df = df[df['approx_cost(for two people)'].notna()]
            df = df[df['rest_type'].notna()]

            logging.info("NULL removed from DataFrame")


            # Splitting clustered values in rest_type into multiple rows
            df['rest_type'] = df['rest_type'].apply(lambda x: str(x).split(','))
            df = df.explode('rest_type')

            logging.info("DataFrame exploded")


            # Dropping duplicates
            df.drop_duplicates(inplace=True)

            logging.info("Duplicates dropped")


            logging.info("preproces() has ended")

            return df
        
        except Exception as e:
            raise CustomException(e, sys)


    # Function to preprocess feature rate
    def prep_rate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        This function changes values 'NEW', '-', np.nan to 0 and
        converts feature 'rate' to float type.

        """
        try:
            logging.info("prep_rate() has begun")

            df['rate'].replace(['NEW', '-', np.nan], 0.0, inplace=True)
            df['rate'] = df['rate'].apply(lambda x: float(str(x)[:3]))

            logging.info("prep_rate() has ended")

            return df
        
        except Exception as e:
            raise CustomException(e, sys)


    # Function to split features and target variable
    def X_y_split(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        This function splits features and target variable into
        X and y and converts target variable into float type.
  
        """
        try:
            logging.info("X_y_split() has begun")

            X: pd.DataFrame = df.drop('approx_cost(for two people)', axis=1)
            y: pd.Series = df['approx_cost(for two people)']

            logging.info("X and y created")

            # Converting 'approx_cost(for two people)' to float
            y = y.apply(lambda x: float(str(x).replace(',','')))
            
            logging.info("y preprocessed")


            logging.info("X_y_split() has ended")

            return X,y
        
        except Exception as e:
            raise CustomException(e, sys)