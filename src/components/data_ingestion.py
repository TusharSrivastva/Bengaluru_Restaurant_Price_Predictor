import os 
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging

from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

from typing import Type, Tuple

@dataclass
class DataIngestionConfig:
    """
    Class DataIngestionConfig initializes the path of train data, test data and original dataset.

        * train_data_path: Path of train dataset
        * test_data_path: Path of test dataset
        * raw_data_path: Path of original dataset

    """
    # Initializing paths for datasets
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')


class DataIngestion:
    """
    class DataIngestion
        * __init__() -> None
        * intiate_data_ingestion() -> Tuple[str, str]

        This class is used for implementing data ingestion, it inherits from class DataIngestionConfig.
        The intiate_data_ingestion() imports the CSV dataset from its locations using pandas library, 
        splits it into train and test datasets and store them in artifacts folder.

    """
    def __init__(self) -> None:
        self.ingestion_config: Type[DataIngestionConfig] = DataIngestionConfig()
    
    def intiate_data_ingestion(self) -> Tuple[str, str]:
        """
        This function implements data ingestion. The CSV dataset is import using pandas library. The 
        dataset is then split into train and test data, further the raw, train and test datasets are 
        stored in the artifacts folder and file paths to train and test datasets are returned as 
        outputs.

        * df: Original dataset
        * train: Training dataset
        * test: Test dataset

        """
        logging.info("Data ingestion has begun")
        try:
            # Importing dataset using pandas
            df: pd.DataFrame = pd.read_csv('notebook\data\zomato.csv')
            logging.info("Dataset imported")

            # Creating directory
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Splitting dataset into test and train
            logging.info("train_test_split initiated")
            train: pd.DataFrame; test: pd.DataFrame
            train, test = train_test_split(df, test_size=0.2, random_state=9)

            # Saving raw, train and test datasets as csv
            df.to_csv(self.ingestion_config.raw_data_path, index = False)
            logging.info("Raw data saved as csv")
            
            train.to_csv(self.ingestion_config.train_data_path, index = False)
            logging.info("Train data saved as csv")

            test.to_csv(self.ingestion_config.test_data_path, index = False)
            logging.info("Test data saved as csv")

            logging.info("Data ingestion complete")  

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )      
       
        except Exception as e:
            raise CustomException(e, sys)
