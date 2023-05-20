import sys
import pandas as pd

from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    """
    class PredictPipelin:
        * __init__() -> None:
        * predict(features) -> int

        This class is used to predict from the given data.
    """
    def __init__(self) -> None:
        pass

    def predict(self, features: pd.DataFrame) -> int:
        """
        * features: Input Dataset
        * model_path: The path to model pickle
        * preprocessor_path: The path to preprocessor pickle
        * model: Pre-trained model
        * preprocessor: Pre-computer preprocessor
        * X_pred: Preprocessed feature dataset
        * y_dummy: Garbage label Series
        * y_pred: Predicted Series

        This function takes a pandas DataFrame as input uses preprocessor pickle
        to preprocess the dataset and uses model model pickle file to predict from 
        the dataset. This function returns the average of all prediction, since a 
        single column is split into multiple columns due to pd.DataFame.explode().
        """
        try:
            # Model and Prepocessor path
            model_path = 'artifacts\model.pkl'
            preprocessor_path = 'artifacts\preprocessor.pkl'

            # Loading model and prepocessor
            model = load_object(file_path = model_path)
            preprocessor = load_object(file_path = preprocessor_path)

            # Preprocessing input data
            X_pred, y_dummy = preprocessor.transform(features)

            # Predicting
            y_pred = model.predict(X_pred)

            return int(y_pred.mean())
        
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    """
    class CustomData:
        * __init__(self,
                 book: str,
                 delivery: str,
                 rate: str,
                 votes: int,
                 location: str,
                 type_tag: str,
                 r_type: str,
                 price:int = 0) -> None
        * get_data_as_dataframe(self) -> pd.DataFrame:

        This class is used to get input from flask and convert that into 
        a pd.DataFrame for prediction

    """
    def __init__(self,
                 book: str,
                 delivery: str,
                 rate: str,
                 votes: int,
                 location: str,
                 type_tag: str,
                 r_type: str,
                 price:int = 0
                 ) -> None:
        """
        * self.book: Table booking ['Yes'|'No']
        * self.delivery: Home delilvery ['Yes'|'No']
        * self.rate: Restaurant rating [0.0, 5.0] 
        * self.votes: Likes received [0, 17000]
        * self.location: Location of the Restaurant
        * self.type_tag: Type tag of the Restaurant 
        * self.r_type: Restaurant type
        * self.price: Price for two [0]
        """
        self.book = book
        self.delivery = delivery
        self.rate = rate 
        self.votes = votes
        self.location = location
        self.type_tag = type_tag 
        self.r_type = r_type
        self.price = price
    
    def get_data_as_dataframe(self) -> pd.DataFrame:
        """
        * custom_data_dict: Dictionary containing data
        * df: DataFrame of the input data

        This function is used to create a dictionary from the input
        data and convert that into a pd.DataFrame.
        """
        try:
            # Creating Dictionary for DataFrame
            custom_data_dict: dict = {
                'online_order': [self.delivery],
                'book_table': [self.book],
                'rate': [self.rate],
                'votes': [self.votes],
                'location': [self.location],
                'rest_type': [self.type_tag],
                'listed_in(type)': [self.r_type],
                'approx_cost(for two people)': [self.price]
            }

            # Creating Dataframe from dictionary
            df: pd.DataFrame = pd.DataFrame(custom_data_dict)

            return df
        
        except Exception as e:
            raise CustomException(e, sys)
