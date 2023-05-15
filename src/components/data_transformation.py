import sys
import os

import pandas as pd

from sklearn import set_config
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer

from src.components.transformation_components.column_transformers import Column_Transformers
from src.components.transformation_components.transformation_functions import Transformation_functions

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

from dataclasses import dataclass

from typing import Tuple, List, Type


@dataclass
class DataTransformationConfig:
    """
    This class is used to intialize path of the preprocessor pickle object.

    * preprocessor_file_path: Contains the path of the preprocessor pickle object.

    """
    # Variable to store preprocessor pickle file path
    preprocessor_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    """
    class DataTransformation:
       * __init__() -> None
       * get_data_transformer_object() -> Pipeline
       * initiate_data_transformation(self,  train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, str]:

       This class inherits DataTransformationConfig class and is used to implement data transformation on the dataset. The get_data_transformer_object()
       initializes a pipeline containing ColumnTransformers and function and passes it to get_data_transformer_object()
       where this pipeline is used to fit_transform the train dataset and transform the test dataset. The 
       get_data_transformer_object() returns X_train, y_train, X_test, y_test and  self.data_tranformation_config.preprocessor_file_path
       as output.

    """
    def __init__(self) -> None:
        self.data_tranformation_config: Type[DataTransformationConfig] = DataTransformationConfig()
    
    def get_data_transformer_object(self) -> Pipeline:
        """
        This function initializes pipeline for data transformation and returns pipleline containing following 
        transformers and functions:

        
        * feature_selection_transformer: This transformer selects important columns from the dataset, drop the rest.
        
        * preprocess_transformer: This transformer preprocesses the dataset using Transformation_functions().preprocess()
        
        * preprocess_rate_transformer: This transformer preprocesses rate column using Transformation_functions().prep_rate()
        
        * impute_rate_transformer: This transformer uses SimpleImputer() to impute rate column using mean as strategy

        * Ordinal_encoder_transformer: This transformer uses OrdinalEncoder() to encode columns online_order, book_table, location, rest_type, listed_in(type)

        * MinMaxScaler_transformer: This transformer scales columns location, rest_type, votes

        * Transformation_functions().X_y_split(): split feature variables and target variable (approx_cost(for two people)) and converts target variable to float type.
        
        """
        try:
            logging.info("Pipeline creation has started")

            # Setting sklearn global configurations
            set_config(transform_output="pandas")

            logging.info("Global connfigurations set")
            
            # List of ColumnTransformers
            List_Column_transformers: List[ColumnTransformer] = Column_Transformers().get_transformers()

            logging.info("List of transformers received")

            # ColumnTransformers for pipeline
            feature_selection_transformer: ColumnTransformer = List_Column_transformers[0]
            preprocess_transformer: ColumnTransformer = List_Column_transformers[1]
            preprocess_rate_transformer: ColumnTransformer = List_Column_transformers[2]
            impute_rate_transformer: ColumnTransformer = List_Column_transformers[3]
            Ordinal_encoder_transformer: ColumnTransformer = List_Column_transformers[4]
            MinMaxScaler_transformer: ColumnTransformer = List_Column_transformers[5]

            logging.info("Transformer variables created")

            # Pipeline for data transformation
            preprocessor: Pipeline = Pipeline(steps=[
                ("Feature_Selection", feature_selection_transformer),
                ("Preprocessing", preprocess_transformer),
                ("Preprocess_rate", preprocess_rate_transformer),
                ("SimpleImpute_rate", impute_rate_transformer),
                ("OrdinalEncode", Ordinal_encoder_transformer),
                ("MinMaxScale", MinMaxScaler_transformer),
                ("X_y_split", FunctionTransformer(Transformation_functions().X_y_split))
            ]
            )

            logging.info("Preprocessor object created")

            logging.info("Pipeline creation has ended")


            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self, 
                                     train_path: str, 
                                     test_path: str) -> Tuple[pd.DataFrame,
                                                              pd.Series,
                                                              pd.DataFrame,
                                                              pd.Series, 
                                                              str]:
        """
        This function is used to perform data transformation on the dataset, and save pickled preprocessor
        object. This function returns X_train, y_train, X_test, y_test and self.data_tranformation_config.preprocessor_file_path.

        * train_df: Training dataset
        * test_df: Test dataset
        * preprocessor_obj: Pipeline for data transformation.
        * X_train: Transformed train feature dataset
        * y_train: Transformed train target series
        * X_test: Transformed test feature dataset
        * y_test: Transformed test series
        * self.data_tranformation_config.preprocessor_file_path: Path of pickled preprocessor object

        """
        try:
            logging.info("intiate_data_transformation() has begun")

            # Importing train and test datasets
            train_df: pd.DataFrame = pd.read_csv(train_path)
            test_df: pd.DataFrame = pd.read_csv(test_path)

            logging.info("Train and test data imported")


            logging.info("Obtaining preprocessor object")
            
            # Getting preprocessor object
            preprocessor_obj: Pipeline = self.get_data_transformer_object()


            logging.info("Starting data transformation")
            
            # Feature dataset and target series
            X_train: pd.DataFrame; y_train: pd.Series
            X_test: pd.DataFrame; y_test: pd.Series

            # Transforming train and test datasets using pipeline object
            X_train, y_train = preprocessor_obj.fit_transform(train_df)
            X_test, y_test = preprocessor_obj.transform(test_df)

            logging.info("Data transformation complete")


            logging.info("Saving preprocessor object")

            # Function to store the preprocessor as pickle
            save_object(
                file_path=self.data_tranformation_config.preprocessor_file_path,
                obj=preprocessor_obj
            )

            logging.info("Saved preprocessor object")


            return(
                X_train,
                y_train,
                X_test,
                y_test,
                self.data_tranformation_config.preprocessor_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)