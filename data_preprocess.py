import pandas as pd
import logging
import sys
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Configure logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fill_missing_values(df: pd.DataFrame, col: str) -> pd.Series:
    """
    Fill missing values in a column based on its data type.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    col (str): The column name in the DataFrame.

    Returns:
    pd.Series: A pandas Series with missing values filled.
    """
    if df[col].dtype == 'object':
        return df[col].fillna("unknown")
    else:
        return df[col].fillna(df[col].mean())

def standardize_data(df: pd.DataFrame, columns: list, scaler: StandardScaler) -> pd.DataFrame:
    """
    Standardize numerical columns in the DataFrame.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    columns (list): List of column names to be standardized.
    scaler (StandardScaler): The StandardScaler instance to use.

    Returns:
    pd.DataFrame: DataFrame with standardized numerical columns.
    """
    df[columns] = scaler.fit_transform(df[columns])
    return df

def encode_categorical_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Encode categorical columns using LabelEncoder.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    columns (list): List of categorical column names.

    Returns:
    pd.DataFrame: The DataFrame with encoded categorical columns.
    """
    le = LabelEncoder()
    for col in columns:
        df[col] = le.fit_transform(df[col])
    return df

def preprocess_and_clean_data(df: pd.DataFrame, standardize: bool = True) -> pd.DataFrame:
    """
    Preprocess and clean a DataFrame with features and individuals.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the data.
    standardize (bool): Whether to standardize numerical features.

    Returns:
    pd.DataFrame: A cleaned and preprocessed DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input is not a pandas DataFrame.")

    try:
        logging.info("Starting data preprocessing and cleaning.")

        df = df.drop_duplicates()

        # Fill missing values
        for col in df.columns:
            df[col] = fill_missing_values(df, col)

        # Standardize numerical columns if needed
        if standardize:
            num_cols = df.select_dtypes(include=['number']).columns
            scaler = StandardScaler()
            df = standardize_data(df, num_cols, scaler)

        # Encode categorical columns
        cat_cols = df.select_dtypes(include=['object']).columns
        df = encode_categorical_columns(df, cat_cols)

        logging.info("Data preprocessing and cleaning completed.")
        return df
    except Exception as e:
        logging.error(f"An error occurred during data preprocessing and cleaning: {str(e)}")
        raise

