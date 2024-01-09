import pandas as pd
import logging
import sys
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Configure logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fill_missing_values(df, col):
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

def standardize_column(df, col, scaler):
    """
    Standardize a numerical column using StandardScaler.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    col (str): The column name to be standardized.
    scaler (StandardScaler): The StandardScaler instance to use.

    Returns:
    pd.Series: A pandas Series with standardized values.
    """
    return scaler.fit_transform(df[col].values.reshape(-1, 1))

def encode_categorical_columns(df):
    """
    Encode categorical columns using LabelEncoder.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.

    Returns:
    pd.DataFrame: The DataFrame with encoded categorical columns.
    """
    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = le.fit_transform(df[col])
    return df

def preprocess_and_clean_data(df, standardize=True):
    """
    Preprocess and clean a DataFrame with features and individuals.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the data.
    standardize (bool): Whether to standardize numerical features.

    Returns:
    pd.DataFrame: A cleaned and preprocessed DataFrame.
    """
    try:
        df = df.drop_duplicates()

        if standardize:
            scaler = StandardScaler()

        for col in df.columns:
            df[col] = fill_missing_values(df, col)
            if df[col].dtype != 'object' and standardize:
                df[col] = standardize_column(df, col, scaler)

        df = encode_categorical_columns(df)

        logging.info("Data preprocessing and cleaning completed.")
        return df
    except Exception as e:
        logging.error(f"An error occurred during data preprocessing and cleaning: {str(e)}")
        raise

