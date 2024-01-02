import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .load_data import load_data

import logging
# Configure logging
logging.basicConfig(filename='data_preprocessing.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def explore_data(df):
    """
    Perform exploratory data analysis (EDA) on the DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the data.

    Returns:
    None
    """
    try:
        # Summary statistics
        logging.info("Summary Statistics:")
        logging.info(df.describe())

        # Data types and missing values
        logging.info("\nData Types and Missing Values:")
        logging.info(df.info())

        # Distribution of categorical variables
        categorical_columns = ["Gender", "Ever_Married", "Graduated", "Profession", "Spending_Score", "Var_1", "Segmentation"]
        for col in categorical_columns:
            logging.info("\nDistribution of %s", col)
            logging.info(df[col].value_counts())

        # Distribution of numerical variables
        numerical_columns = ["Age", "Work_Experience", "Family_Size"]
        for col in numerical_columns:
            plt.figure(figsize=(8, 5))
            sns.histplot(df[col], kde=True)
            plt.title(f"Distribution of {col}")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plt.show()

        # Correlation heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm", linewidths=0.5)
        plt.title("Correlation Heatmap")
        plt.show()

        # Box plots for numerical variables by Segmentation
        for col in numerical_columns:
            plt.figure(figsize=(8, 5))
            sns.boxplot(x="Segmentation", y=col, data=df)
            plt.title(f"{col} by Segmentation")
            plt.xlabel("Segmentation")
            plt.ylabel(col)
            plt.show()

        # Countplot for categorical variables by Segmentation
        for col in categorical_columns[:-1]:  # Exclude "Segmentation" from categorical columns
            plt.figure(figsize=(8, 5))
            sns.countplot(x=col, hue="Segmentation", data=df)
            plt.title(f"{col} by Segmentation")
            plt.xlabel(col)
            plt.ylabel("Count")
            plt.legend(title="Segmentation")
            plt.show()

    except Exception as e:
        logging.error(f"An error occurred during data exploration: {str(e)}")
        raise Exception(f"An error occurred during data exploration: {str(e)}")

if __name__ == "__main__":
    try:
        # Load the dataset
        df = load_data("sample_data/Train.csv")

        # Perform exploratory data analysis (EDA)
        explore_data(df_clean_preprocessed)
    except Exception as e:
        logging.error(f"Error: {str(e)}")
