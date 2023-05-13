import os 
import sys
from src.exception import CustomException
from src.logger import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join('artifacts', 'train.csv')
    test_data_path = os.path.join('artifacts', 'test.csv')
    raw_data_path = os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def intiate_data_ingestion(self)->str:
        logging.info("Data ingestion has begun")
        try:
            # Importing dataset using pandas
            df = pd.read_csv('notebook\data\zomato.csv')
            logging.info("Dataset imported")

            # Creating directory
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Splitting dataset into test and train
            logging.info("train_test_split initiated")
            train, test = train_test_split(df, test_size=0.2, random_state=9)

            # Saving datasets as csv
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

if __name__=="__main__":
    obj = DataIngestion()
    obj.intiate_data_ingestion()

