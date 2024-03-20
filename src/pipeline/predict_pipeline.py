import sys
import pandas as pd

from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self, features):
        try:
            model_path = r'C:\BlackFriday\artifacts\model.pkl'
            preprocessor_path = r"C:\BlackFriday\artifacts\preprocessor.pkl"
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
        
        except Exception as e:
            raise CustomException(e, sys)
        
        return preds
    
    
    
class CustomData:
    def __init__( self,
        Gender: str,
        Age: str,
        Occupation: int,
        City_Category: str,
        Stay_In_Current_City_Years: int,
        Marital_Status: int,
        Product_Category_1: int,
        Product_Category_2: int,
        Product_Category_3: int):
        
        self.Gender = Gender
        
        self.Age = Age
        
        self.Occupation = Occupation
        
        self.City_Category = City_Category
        
        self.Stay_In_Current_City_Years = Stay_In_Current_City_Years
        
        self.Marital_Status = Marital_Status
        
        self.Product_Category_1 = Product_Category_1
        
        self.Product_Category_2 = Product_Category_2
        
        self.Product_Category_3 = Product_Category_3
        
    def get_data_as_data_frame(self):
        
        try:
            custome_data_input_dict = {
                "Gender": [self.Gender],
                "Age": [self.Age],
                "Occupation": [self.Occupation],
                "City_Category": [self.City_Category],
                "Stay_In_Current_City_Years": [self.Stay_In_Current_City_Years],
                "Marital_Status": [self.Marital_Status],
                "Product_Category_1": [self.Product_Category_1],
                "Product_Category_2": [self.Product_Category_2],
                "Product_Category_3": [self.Product_Category_3]
            }
            return pd.DataFrame(custome_data_input_dict)
        
        
        except Exception as e:
            raise CustomException(e, str)