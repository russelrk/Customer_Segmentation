import argparse
from load_data import load_data
from data_preprocess import preprocess_and_clean_data
from logistic_regression import train_logistic_regression_model
from eda_analysis import explore_data

def main(data_path):
    # Load your data
    df = load_data(data_path)

    # EDA
    explore_data(df)

    # Preprocess and clean the data
    df_cleaned = preprocess_and_clean_data(df)

    # Train the logistic regression model
    classification_report_result = train_logistic_regression_model(df_cleaned)
    print(classification_report_result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perform EDA and logistic regression.')
    parser.add_argument('data_path', type=str, help='Path to the data file (e.g., sample_data/Train.csv)')
    args = parser.parse_args()
    
    data_path = args.data_path
    main(data_path)
