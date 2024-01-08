import pandas as pd
import logging
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler  # Import the StandardScaler

# Configure logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def preprocess_and_clean_data(df, standardize=True):
    """
    Preprocess and clean a DataFrame with 10 features and 8000 individuals.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the data.
    standardize (bool): Whether to standardize numerical features.

    Returns:
    pd.DataFrame: A cleaned and preprocessed DataFrame.
    """
    try:
        # Remove duplicates if any
        df = df.drop_duplicates()

        # Initialize the scaler
        scaler = StandardScaler()

        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna("unknown")
            else:
                # Calculate the mean for non-object columns
                df[col] = df[col].fillna(df[col].mean())
                if standardize:
                    # Standardize numerical columns
                    df[col] = scaler.fit_transform(df[col].values.reshape(-1, 1))

        # Create a label encoder object
        le = LabelEncoder()

        # Apply the label encoder to the DataFrame for object columns
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = le.fit_transform(df[col])

        # Other preprocessing steps can be added as needed

        logging.info("Data preprocessing and cleaning completed.")
        return df
    except Exception as e:
        logging.error(f"An error occurred during data preprocessing and cleaning: {str(e)}")
        raise Exception(f"An error occurred during data preprocessing and cleaning: {str(e)}")
