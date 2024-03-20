import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    """Load the Black Friday dataset."""
    return pd.read_csv(file_path)

def basic_overview(df):
    """Get a basic overview of the dataset."""
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns}")
    print(df.info())
    print(df.describe())

def missing_value_analysis(df):
    """Analyze missing values in the dataset."""
    print("Missing values per column:")
    print(df.isnull().sum())
    sns.heatmap(df.isnull(), cbar=False)
    plt.title("Missing Value Heatmap")
    plt.show()

def categorical_analysis(df, categorical_cols):
    """Analyze categorical variables."""
    for col in categorical_cols:
        print(f"Unique values in {col}: {df[col].unique()}")
        print(f"Value counts for {col}:")
        print(df[col].value_counts())
        sns.countplot(data=df, x=col)
        plt.show()

def numerical_analysis(df, numerical_cols):
    """Analyze numerical variables."""
    for col in numerical_cols:
        sns.histplot(data=df, x=col, kde=True)
        plt.title(f"Distribution of {col}")
        plt.show()
        sns.boxplot(data=df, x=col)
        plt.title(f"Boxplot of {col}")
        plt.show()

def correlation_analysis(df, numerical_cols):
    """Analyze correlations between numerical variables."""
    sns.pairplot(df[numerical_cols])
    plt.show()
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[numerical_cols].corr(), annot=True, cmap="YlGnBu")
    plt.title("Correlation Heatmap")
    plt.show()

# Usage example
file_path = "black_friday.csv"
data = load_data(file_path)
basic_overview(data)
missing_value_analysis(data)
categorical_cols = ["Product_Category_1", "Product_Category_2", "Product_Category_3"]
categorical_analysis(data, categorical_cols)
numerical_cols = ["Purchase", "Age", "City_Population"]
numerical_analysis(data, numerical_cols)
correlation_analysis(data, numerical_cols)