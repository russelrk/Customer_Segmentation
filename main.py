import argparse
import logging
from load_data import load_data
from data_preprocess import preprocess_and_clean_data
from logistic_regression import train_logistic_regression_model
from eda_analysis import explore_data

def main(data_path: str, eda: bool = True, cl_report: bool = True) -> None:
    """
    Main function to perform EDA and logistic regression on the provided dataset.
    
    :param data_path: Path to the data file.
    :param eda: Flag to perform EDA (Exploratory Data Analysis).
    :param cl_report: Flag to generate a classification report.
    """
    # Load your data
    df = load_data(data_path)
    
    try:
        if eda:
            # EDA
            explore_data(df)

    except Exception as e:
        logging.error(f"Error occurred: {e}")

    try:     
        if cl_report:
            # Preprocess and clean the data
            df_cleaned = preprocess_and_clean_data(df)
        
            # Train the logistic regression model
            classification_report_result = train_logistic_regression_model(df_cleaned)
            logging.info(classification_report_result)
    except Exception as e:
        logging.error(f"Error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perform EDA and logistic regression.')
    parser.add_argument('data_path', type=str, help='Path to the data file (e.g., sample_data/Train.csv)')
    parser.add_argument('--eda', dest='eda', action='store_true', help='Enable EDA (Exploratory Data Analysis)')
    parser.add_argument('--no-eda', dest='eda', action='store_false', help='Disable EDA')
    parser.add_argument('--cl_report', dest='cl_report', action='store_true', help='Generate a classification report')
    parser.add_argument('--no-cl_report', dest='cl_report', action='store_false', help='Do not generate a classification report')
    parser.add_argument('--log_level', type=str, default='INFO', help='Set the logging level (e.g., INFO, DEBUG)')
    parser.set_defaults(eda=True, cl_report=True)

    args = parser.parse_args()
    
    logging.basicConfig(level=args.log_level.upper())
    
    main(args.data_path, args.eda, args.cl_report)
