import pandas as pd
import logging
from .load_data import load_data

# Configure logging
logging.basicConfig(filename='data_preprocessing.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_path):
    """
    Load the dataset from a CSV file.

    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    pd.DataFrame: The loaded DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
        logging.info("Data loaded successfully.")
        return df
    except FileNotFoundError:
        logging.error(f"File not found at '{file_path}'")
        raise FileNotFoundError(f"File not found at '{file_path}'")
    except Exception as e:
        logging.error(f"An error occurred while loading data: {str(e)}")
        raise Exception(f"An error occurred while loading data: {str(e)}")

def preprocess_and_clean_data(df):
    """
    Preprocess and clean a DataFrame with 10 features and 8000 individuals.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the data.

    Returns:
    pd.DataFrame: A cleaned and preprocessed DataFrame.
    """
    try:
        # Remove duplicates if any
        df = df.drop_duplicates()

        # Handle missing values (you can customize this based on your data)
        df = df.fillna(df.mean(numeric_only=True))  # Fill missing values with mean

        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna('Unknown')

        # Remove outliers (you can customize this based on your data)
        # Example: Remove rows where a specific feature is beyond a certain threshold
        # df = df[df['feature_name'] < threshold]

        # Encode categorical variables if needed (e.g., one-hot encoding)
        # Example: df = pd.get_dummies(df, columns=['categorical_feature'])

        # Standardize/normalize numerical features if needed
        # Example: df['numeric_feature'] = (df['numeric_feature'] - df['numeric_feature'].mean()) / df['numeric_feature'].std()

        # Feature engineering (create new features if needed)

        # Drop unnecessary columns if any
        # Example: df.drop(['unnecessary_feature1', 'unnecessary_feature2'], axis=1, inplace=True)

        logging.info("Data preprocessing and cleaning completed.")
        return df
    except Exception as e:
        logging.error(f"An error occurred during data preprocessing and cleaning: {str(e)}")
        raise Exception(f"An error occurred during data preprocessing and cleaning: {str(e)}")

if __name__ == "__main__":
    try:
        # Load the dataset
        df_train = load_data("customer_segment_data/Train.csv")

        # Preprocess and clean the data
        df_cleaned_preprocessed = preprocess_and_clean_data(df_train.copy())

        # You can further process or analyze the cleaned data here or save it to a new file.
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
