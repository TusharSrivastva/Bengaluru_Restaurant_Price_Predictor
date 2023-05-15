import pandas as pd
import numpy as np
import sys

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler

from src.components.transformation_components.transformation_functions import Transformation_functions

from src.exception import CustomException
from src.logger import logging

from typing import List

class Column_Transformers:
    """
    class Column_Transformers:
        * __init__() -> None
        * get_transformers() -> List[ColumnTransformer]

        This class is used to initialize ColumnTransformers for Data Transformation.
        The get_transformers() initializes ColumnTransfomers and returns a list of 
        ColumnTransformers.

    """
    def __init__(self) -> None:
        pass

    def get_transformers(self) -> List[ColumnTransformer]:
        """
        This function initializes ColumnTransforms and creates a list of 
        them to return.

        * select_trf: Select important feature columns and drop the rest
        * pre_trf: Preprocess dataset using Transformation_functions().preprocess() for further transformation
        * pre_rate_trf: Preprocess rate column using Transformation_functions().prep_rate()
        * imp_rate_trf: Impute rate column using SimpleImputer()
        * encode_trf: Encode some columns using OrdinalEncoder
        * scale_trf: Scale some columns using MinMaxScale()
        * List_Column_transformers: List of all the ColumnTransformers
        
        """
        try:
            logging.info("get_transformers() has begun")

            # Transformer to select important features
            select_trf: ColumnTransformer = ColumnTransformer(transformers=[
                ("SelectColumns", 'passthrough', ['online_order',
                                                  'book_table',
                                                  'rate',
                                                  'votes',
                                                  'location',
                                                  'rest_type',
                                                  'approx_cost(for two people)',
                                                  'listed_in(type)'])
            ], verbose_feature_names_out=False)

            logging.info("select_trf created")


            # Transformer to do necessary preprocessing
            pre_trf: ColumnTransformer = ColumnTransformer(transformers=[
                ("Preprocess", FunctionTransformer(Transformation_functions().preprocess), slice(0,8))
            ], verbose_feature_names_out=False)

            logging.info("pre_trf created")


            # Transformer to preprocess 'rate' column
            pre_rate_trf: ColumnTransformer = ColumnTransformer(transformers=[
                ("PreprocessRate", FunctionTransformer(Transformation_functions().prep_rate), slice(0,8))
            ], verbose_feature_names_out=False)

            logging.info("pre_rate_trf created")


            # Transformer to impute 'rate' column
            imp_rate_trf: ColumnTransformer = ColumnTransformer(transformers=[
                ("SimpleImputer", SimpleImputer(missing_values=0.0, strategy="mean"), ['rate'])
            ], remainder = 'passthrough', verbose_feature_names_out=False)

            logging.info("imp_rate_trf created")


            # Transformer to Ordinal Encode columns
            encode_trf: ColumnTransformer = ColumnTransformer(transformers=[
                ("OrdinalEncoder", OrdinalEncoder(), ['online_order',
                                                      'book_table',
                                                      'location',
                                                      'rest_type',
                                                      'listed_in(type)'])
            ], remainder = 'passthrough', verbose_feature_names_out=False)

            logging.info("encode_trf created")


            # Transformer to MinMax scale columns
            scale_trf: ColumnTransformer = ColumnTransformer(transformers=[
                ("MinMaxScaler", MinMaxScaler(), ['location',
                                                  'rest_type',
                                                  'votes'])
            ], remainder = 'passthrough', verbose_feature_names_out=False)

            logging.info("scale_trf created")

            # Creating a list of all the ColumnTransformers to return
            List_Column_transformers: List[ColumnTransformer] = [select_trf,
                                                                 pre_trf,
                                                                 pre_rate_trf,
                                                                 imp_rate_trf,
                                                                 encode_trf,
                                                                 scale_trf]
            logging.info("List of Column Transformers complete")


            logging.info("get_transformers() complete")
            

            return List_Column_transformers
        
        except Exception as e:
            raise CustomException(e, sys)
