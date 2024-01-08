import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .load_data import load_data
import logging
import sys

# Configure logging to write to stdout
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
            plt.savefig(f"{col}_distribution.png")  # Save the plot as an image
            plt.close()  # Close the plot to prevent displaying it in the Docker container output

        # Correlation heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm", linewidths=0.5)
        plt.title("Correlation Heatmap")
        plt.savefig("correlation_heatmap.png")  # Save the plot as an image
        plt.close()  # Close the plot to prevent displaying it in the Docker container output

        # Box plots for numerical variables by Segmentation
        for col in numerical_columns:
            plt.figure(figsize=(8, 5))
            sns.boxplot(x="Segmentation", y=col, data=df)
            plt.title(f"{col} by Segmentation")
            plt.xlabel("Segmentation")
            plt.ylabel(col)
            plt.savefig(f"{col}_boxplot.png")  # Save the plot as an image
            plt.close()  # Close the plot to prevent displaying it in the Docker container output

        # Countplot for categorical variables by Segmentation
        for col in categorical_columns[:-1]:  # Exclude "Segmentation" from categorical columns
            plt.figure(figsize=(8, 5))
            sns.countplot(x=col, hue="Segmentation", data=df)
            plt.title(f"{col} by Segmentation")
            plt.xlabel(col)
            plt.ylabel("Count")
            plt.legend(title="Segmentation")
            plt.savefig(f"{col}_countplot.png")  # Save the plot as an image
            plt.close()  # Close the plot to prevent displaying it in the Docker container output

    except Exception as e:
        logging.error(f"An error occurred during data exploration: {str(e)}")
        raise Exception(f"An error occurred during data exploration: {str(e)}")
