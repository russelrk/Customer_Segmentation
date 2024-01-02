import pandas as pd
import logging

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
