# data_cleaning_module.py
import os
import shutil

import pandas as pd
import numpy as np
from scipy.stats import percentileofscore, mstats
from scipy.stats.mstats import winsorize

import matplotlib.pyplot as plt



import warnings
warnings.filterwarnings("ignore")

"""
Function to load data from a specified file path.
It uses pandas to read a CSV file and returns a pandas DataFrame.
"""
def load_data(file_path):
    """Load the Black Friday dataset."""
    return pd.read_csv(file_path, dtype={'Age': str})


"""
Function to print a basic overview of the DataFrame.
It displays the columns, the shape of the DataFrame, 
general information, and statistics of the data.
"""
def basic_overview(df):
    
    #Get a basic overview of the DataFrame.
    print(f"Columns: {df.columns.values.tolist()}")
    print("")
    print("The dataset shape")
    print(f"Shape: {df.shape}")
    print("")
    print(df.info())
    print("")
    print(df.describe(include="all").T)
    
    return df

    
"""
Function to analyze and print the missing values in the DataFrame.
It calculates the total number of missing values per column and 
the percentage of missing values.
"""
def missing_value_analysis(df):
    
    # Analyze missing values in the dataset.
    print("")
    missing_values_per_column = df.isnull().sum()
    print("Missing values per column:")
    print(missing_values_per_column)

    print("")
    # Percentage of NaN for the dataset
    print("Missing values per column in %:")
    missing_values_percentage = df.isna().mean().round(4).mul(100).sort_values(ascending=False)
    print(missing_values_percentage)

    # Return the missing value information
    return missing_values_per_column, missing_values_percentage


"""
Function to check for duplicate rows in the DataFrame.
It identifies duplicate rows and returns a boolean indicating i
f duplicates were found, along with the duplicate rows themselves.
"""
def check_duplicates(df):
    
    # Check for duplicate rows in a DataFrame.
    print("")
    duplicates = df[df.duplicated(keep=False)]
    if not duplicates.empty:
        print("Duplicates found:")
        print(duplicates)
        return True, duplicates
    else:
        print("No duplicates found.")
        
        return False, None
    
    
"""
Function to check for and remove duplicate rows in the DataFrame.
It identifies and removes duplicates, then returns the deduplicated 
DataFrame and a boolean indicating if duplicates were removed.
"""    
def check_and_remove_duplicates(df):
    
    # Check for duplicate rows in a DataFrame and remove them if they exist.
    print("")
    duplicates = df[df.duplicated(keep=False)]
    if not duplicates.empty:
        print("Duplicates found:")
        print(duplicates)
        
        # Remove duplicates
        df_deduped = df.drop_duplicates()
        print(f"Removed {len(duplicates)} duplicate rows.")
        
        return df_deduped, True
    else:
        print("No duplicates found.")
        
        return df, False   
    

"""
This function identifies outliers in a specified column of a pandas DataFrame 
using the Interquartile Range (IQR) method, then calculates and returns the 
percentage of these outliers. 
It returns a DataFrame of outliers, their indices, and their percentage relative 
to the total dataset.
"""
def check_for_outliers(df, column_name):
    
    # Check for outliers in a specified column of a pandas DataFrame using the IQR method, and calculate the percentage of outliers.
    
    # Calculate Q1 (25th percentile) and Q3 (75th percentile) of the column
    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    
    # Calculate the Interquartile Range (IQR)
    IQR = Q3 - Q1
    
    # Define the boundaries for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    print("")
    print(f"The Lower Boundary is: {lower_bound}")
    print(f"The Upper Boundary is: {upper_bound}")
    
    # Find outliers
    outliers = df[(df[column_name] < lower_bound) | (df[column_name] > upper_bound)]
    # print(f"The outlier is: {outliers}")
    outlier_indices = outliers.index.tolist()
    
    # Calculate the percentage of outliers
    total_entries = df.shape[0]
    outlier_count = len(outliers)
    outlier_percentage = (outlier_count / total_entries) * 100
    
    # Report findings
    if not outliers.empty:
        print(f"Outliers detected in '{column_name}': {len(outliers)} instances, which is {outlier_percentage:.2f}% of the dataset.")
    else:
        print(f"No outliers detected in '{column_name}'.")
    print("")
    
    return outliers, outlier_indices, outlier_percentage




def refined_dynamic_winsorize(df, column_name):
    """
    Dynamically handle outliers in the specified column of a pandas DataFrame using the Winsorize method,
    based on the distribution of the data, and plot a boxplot after winsorization to visualize the effect.
    """
    # Calculate the initial number of outliers
    initial_outliers = df[(df[column_name] < df[column_name].quantile(0.01)) | 
                          (df[column_name] > df[column_name].quantile(0.99))]
    initial_outlier_count = len(initial_outliers)

    print(f"Initial outliers detected in '{column_name}': {initial_outlier_count} instances.")

    if initial_outlier_count > 0:
        # Apply winsorization based on the 1st and 99th percentiles
        winsorized_data = winsorize(df[column_name], limits=[0.01, 0.01])
        df[column_name] = winsorized_data
        
        # Calculate and report new outlier count after winsorization
        new_outliers = df[(df[column_name] < np.percentile(winsorized_data, 1)) | 
                          (df[column_name] > np.percentile(winsorized_data, 99))]
        new_outlier_count = len(new_outliers)
        
        print(f"Outliers after winsorization in '{column_name}': {new_outlier_count} instances.")
    else:
        print("No significant outliers detected; no Winsorizing applied.")
    
    # # Plot boxplot after winsorization
    # plt.figure(figsize=(6, 4))
    # plt.boxplot(df[column_name], vert=False)
    # plt.title(f'After Winsorization: {column_name}')
    # plt.show()
    
    return df

    
"""
Function to find and print unique values for specified columns in the DataFrame.
It is designed to work with columns related to product categories in this context.
"""
def check_unique_value(df):
    
    # Find and print unique values for specified columns.
    print("")
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input is not a pandas DataFrame")
    
    # Finding unique values for specified columns
    unique_values_pc1 = df['Product_Category_1'].unique()
    unique_values_pc2 = df['Product_Category_2'].unique()
    unique_values_pc3 = df['Product_Category_3'].unique()

    # Printing the unique values
    print("Unique values in Product_Category_1:", unique_values_pc1)
    print("Unique values in Product_Category_2:", unique_values_pc2)
    print("Unique values in Product_Category_3:", unique_values_pc3)

    # Return the DataFrame
    return df


"""
Function to handle missing values in the DataFrame according to a specified 
strategy (mean, median, mode, or a specific value).
It fills the missing values accordingly and returns the modified DataFrame.
"""

# def handle_missing_values(df, strategy='mean'):
    
#     # Handle missing values in a DataFrame.
#     print("")
#     if strategy == 'mean':
#         return df.fillna(df.mean())
#     elif strategy == 'median':
#         return df.fillna(df.median())
#     elif strategy == 'mode':
#         return df.fillna(df.mode().iloc[0])
#     else:
#         return df.fillna(strategy)


def handle_missing_values(df, strategy='mean', inplace=False):
    """
    Handle missing values in a DataFrame based on a specified strategy.

    Parameters:
    df (DataFrame): The DataFrame to process.
    strategy (str or scalar): Strategy for filling missing values ('mean', 'median', 'mode', or a scalar value).
    inplace (bool): If True, fill missing values in place. Otherwise, return a copy of the DataFrame.

    Returns:
    DataFrame: The DataFrame with missing values handled, unless inplace=True.
    """
    if not inplace:
        df = df.copy()

    if strategy == 'mean':
        df.fillna(df.mean(), inplace=True)
    elif strategy == 'median':
        df.fillna(df.median(), inplace=True)
    elif strategy == 'mode':
        # Fill with the first mode if multiple modes exist
        df.fillna(df.mode().iloc[0], inplace=True)
    else:
        # Assume strategy is a scalar fill value
        df.fillna(strategy, inplace=True)

    if inplace:
        return None
    else:
        return df

    
    
def remove_unnecessary_columns(df, columns_to_remove):
    
    # Remove specified columns from a DataFrame.
    
    # Ensure all columns_to_remove are actually in the DataFrame
    columns_to_remove = [column for column in columns_to_remove if column in df.columns]
    
    # Drop the unnecessary columns
    data_cleaned = df.drop(columns=columns_to_remove, axis=1)
    
    return data_cleaned




def convert_columns_to_int(df, column_names):
    """
    Convert specified columns in a DataFrame from float to int.
    
    Parameters:
    df (DataFrame): The DataFrame containing the columns to convert.
    column_names (list): A list of names of the columns to convert to integers.
    
    Returns:
    DataFrame: The DataFrame with the converted columns.
    """
    for column_name in column_names:
        # Check if the column exists in the DataFrame
        if column_name not in df.columns:
            print(f"Column '{column_name}' not found in DataFrame.")
            continue  # Skip this column and move to the next

        # Check if the column can be converted to int
        if not pd.api.types.is_float_dtype(df[column_name]):
            print(f"Column '{column_name}' is not of float type or has NaN values that cannot be converted.")
            continue  # Skip this column and move to the next
        
        # Convert the column to int, assuming NaN values have been appropriately handled before this step
        try:
            df[column_name] = df[column_name].astype(int)
            print(f"Column '{column_name}' successfully converted to int.")
        except ValueError as e:
            print(f"Error converting column '{column_name}' to int: {e}")
    return df






"""
# Main preprocessing function that combines all the above functions to clean and preprocess the data.
# It loads the data, performs an overview, analyzes missing values, 
# checks for unique values, removes duplicates, handles missing values, check and fix outlier,
# remove unnecessary column and returns the cleaned DataFrame.
"""
def preprocess_data(file_path):
    
    # Load the data
    data = load_data(file_path)
    
    # View basic overview
    basic_overview(data)
    
    # Missing value analysis
    missing_value_analysis(data)
    
    # Handle missing value
    data = handle_missing_values(data, strategy='mode')
    
     # Define columns to remove
    columns_to_remove = ['User_ID', 'Product_ID']
    # Remove unnecessary columns
    data = remove_unnecessary_columns(data, columns_to_remove)
    
    columns_to_convert_to_int = ['Product_Category_2', 'Product_Category_3']
    data = convert_columns_to_int(data, columns_to_convert_to_int)
    
    
    
    # Check unique value
    check_unique_value(data)
    
    # Check for duplicates
    has_duplicates, duplicate_rows = check_duplicates(data)

    # Check and remove duplicate values if necessary
    if has_duplicates:
        data, duplicates_removed = check_and_remove_duplicates(data)
    
    # Check for outliers in the 'Purchase' column
    # _, _, outlier_percentage = check_for_outliers(data, 'Purchase')
    
    # # Dynamically Winsorize outliers in the 'Purchase' column
    # data = refined_dynamic_winsorize(data, 'Purchase')
    
    # #again, to verify
    #  # Check for outliers in the 'Purchase' column
    # _, _, outlier_percentage = check_for_outliers(data, 'Purchase')

    
    return data

    
    
    
# Load and preprocess the data
file_path = r'C:\mlprojects\BlackFriday\data\BlackFriday.csv'
cleaned_data = preprocess_data(file_path)

# verify the codes.
print(cleaned_data.head())
print("")
print(cleaned_data.info())
print("")
print(cleaned_data.isnull().sum())
print("")
print(f"Shape: {cleaned_data.shape}")

# Save the clean dataset
cleaned_data_file_path = 'cleaned_black_friday_data.csv'
cleaned_data.to_csv(cleaned_data_file_path, index=False)

# Move the saved file to the data directory
destination_dir = os.path.join(r'C:\mlprojects\BlackFriday\data')

# Create the destination directory if it doesn't exist
os.makedirs(destination_dir, exist_ok=True)

shutil.move(cleaned_data_file_path, destination_dir)



# save the clean dataset
# cleaned_data.to_csv('winsorize_cleaned_black_friday_data.csv', index=False)
# cleaned_data.to_csv('cleaned_black_friday_data.csv', index=False)
