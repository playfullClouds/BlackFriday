import os
import sys

import dill

from src.exception import CustomException
from sklearn.metrics import r2_score

def save_object(file_path, obj):
    
    try:
        dir_path = os.path.dirname(file_path)
        
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    
    
    
    except Exception as e:
        raise CustomException(e, sys)
    
    
def evaluate_models(X_train, y_train, X_validate, y_validate, models):
    try:
        report = {}
        
        for model_name, model in models.items():
        # for i in range(len(list(models))):
            
        #     model = list(models.values())[i]
            
            model.fit(X_train, y_train)
            
            y_train_pred = model.predict(X_train)
            y_validate_pred = model.predict(X_validate)           
            
            train_model_score = r2_score(y_train, y_train_pred)
            validation_model_score = r2_score(y_validate, y_validate_pred)
            
            # report[model_name] = {
            #     'train_score': train_model_score,
            #     'validation_score': validation_model_score
            # }
            
            report[f'{model_name}_train_score'] = train_model_score
            report[f'{model_name}_validation_score'] = validation_model_score
            
            # report[list(models.keys())[1]] = validation_model_score

        return report
           
    except Exception as e:
        raise CustomException(e, sys) 


def load_object(file_path):
    
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
        
    except Exception as e:
        raise CustomException(e, sys)