import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor
)

from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models



@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")
    
   
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def get_best_model_from_report(self, model_report):
        # Extract only validation scores and model names
        validation_scores = {key.replace('_validation_score', ''): score for key, score in model_report.items() if '_validation_score' in key}
       
        # Find the model with the highest validation score
        best_model_name = max(validation_scores, key=validation_scores.get)
        best_model_score = validation_scores[best_model_name]
        
        return best_model_name, best_model_score
    
    def initiate_model_trainer(self, train_array, validate_array):
        
        try:
            logging.info("Split training, validation, and test input data")
            
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_validate, y_validate = validate_array[:, :-1], validate_array[:, -1]

            models = {
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False)
            }
            
            model_report = evaluate_models(X_train=X_train, y_train=y_train, X_validate=X_validate, 
                                           y_validate=y_validate,models=models)
            
            
            # Use the utility function to get the best model based on validation scores
            best_model_name, best_model_score = self.get_best_model_from_report(model_report)
            
            
            if best_model_score < 0.6:
                raise CustomException("No best model found with sufficient score")
            
            logging.info(f"Best model: {best_model_name} with R^2 score: {best_model_score}")

            # Access the best model using its name
            best_model = models[best_model_name]
            best_model.fit(X_train, y_train)  # Ensure the best model is retrained on the full training set
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            predicted = best_model.predict(X_validate)
            r2_square = r2_score(y_validate, predicted)
            
            return r2_square, best_model_name
            
        
        except Exception as e:
            raise CustomException(e, sys)
 