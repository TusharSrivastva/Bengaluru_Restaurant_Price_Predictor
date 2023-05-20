from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

from src.logger import logging


if __name__=="__main__":

    # Data Ingestion
    logging.info("Train_Pipeline: Data Ingestion has begun")
    train_data_path, test_data_path = DataIngestion().intiate_data_ingestion()

    # Data Transformation
    logging.info("Train_Pipeline: Data Transformation has begun")
    X_train, y_train, X_test, y_test, preprocessor_path = DataTransformation().initiate_data_transformation(train_data_path, test_data_path)

    # Model Training
    logging.info("Train_Pipeline: Model Training has begun")
    r2_score, model_path = ModelTrainer().initiate_model_training(X_train=X_train,
                                                                  y_train=y_train,
                                                                  X_test=X_test,
                                                                  y_test=y_test)
    
    logging.info("Train_Pipeline: End")