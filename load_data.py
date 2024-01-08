import pandas as pd
import logging
import sys

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
        sys.exit(1)  # Exit with a non-zero code to indicate failure
    except Exception as e:
        logging.error(f"An error occurred while loading data: {str(e)}")
        sys.exit(1)  # Exit with a non-zero code to indicate failure
