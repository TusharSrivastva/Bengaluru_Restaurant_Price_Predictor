import sys
from src.logger import logging

def error_info(error, error_details: sys)-> str:
    _, _, error_tb = error_details.exc_info()
    file_name: str = error_tb.tb_frame.f_code.co_filename
    error_message: str = "The error occured in python script name [{0}] at line [{1}], error message: [{2}]".format(
        file_name, error_tb.tb_lineno, str(error))
    
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_details: sys) -> None:
        super().__init__(error_message)
        self.error_message = error_info(error_message, error_details)
    
    def __str__(self):
        return self.error_message
    
