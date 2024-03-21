import os  # Import the OS module for interacting with the operating system
import sys  # Import the sys module 
from src.exception import CustomException  # Import CustomException from the exception module stored in the src dir.
from src.logger import logging  # Import the logging object from the logger module for logging
import pandas as pd  # Import the pandas library for data manipulation and analysis

from sklearn.model_selection import train_test_split  # Import train_test_split function for splitting data arrays into two subsets
from dataclasses import dataclass  # Import dataclass decorator for creating data classes

from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig, ModelTrainer

# Decorator to automatically generate special methods for the class below
@dataclass
class DataIngestionConfig:  # Define a data class for data ingestion configuration
    train_data_path: str = os.path.join('artifacts', "train.csv")  # Path to the training data CSV file
    test_data_path: str = os.path.join('artifacts', "test.csv")  # Path to the test data CSV file
    val_data_path: str = os.path.join('artifacts', "val.csv")  # Path to the validation data CSV file
    raw_data_path: str = os.path.join('artifacts', "raw.csv")  # Path to the raw data CSV file

class DataIngestion:  # Define the DataIngestion class
    def __init__(self):  # Constructor method to initialize the DataIngestion object
        self.ingestion_config = DataIngestionConfig()  # Initialize the data ingestion configuration
        
    def initiate_data_ingestion(self):  # Method to start the data ingestion process
        logging.info("Entered the data ingestion method or component")  # Log entry
        try:  # Try block to catch exceptions
            df = pd.read_csv(r'C:\BlackFriday\data\cleaned_black_friday_data.csv')  # Read the dataset into a pandas DataFrame
            logging.info('Read the dataset as dataframe')  # Log entry
            
            # Create directories if they don't exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            
            # Save the DataFrame to the raw CSV file
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            
            
            logging.info("Train test Validate split initiated") # Log entry
            # Split the data into training and temporary sets
            train_set, temp_set = train_test_split(df, test_size=0.4, random_state=42)
            # Split the temporary set into validation and test sets
            val_set, test_set = train_test_split(temp_set, test_size=0.5, random_state=42)
            
            # Save the training set to CSV file
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            
            # Save the test set to CSV file
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            
            # Save the validation set to CSV file
            val_set.to_csv(self.ingestion_config.val_data_path,index=False,header=True)

            logging.info("Ingestion of the data is completed") # Log entry
            
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path, 
                self.ingestion_config.val_data_path
            ) # Return paths to the training, test, and validation data files
        
        except Exception as e: # Catch exceptions
            raise CustomException(e, sys) # Raise CustomException with the caught exception and sys module
        

# Check if the script is run as the main program       
if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data, val_data = obj.initiate_data_ingestion()
    
    data_transformation = DataTransformation()
    train_arr, test_arr, val_arr, preprocessor_obj_file_path = data_transformation.initiate_data_transformation(train_data, test_data, val_data)

    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr, val_arr))