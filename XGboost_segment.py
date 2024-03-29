import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from xgboost import XGBClassifier  # Import XGBClassifier
from load_data import load_data
from data_preprocess import preprocess_and_clean_data

def train_xgboost_model(data_path):
    # Load your data
    # Preprocess and clean the data
    df = preprocess_and_clean_data(load_data(data_path))

    # Assuming df is your DataFrame, split the data into features and labels
    ID, y, X = df.pop('ID'), df.pop('Segmentation'), df

    # Preprocessing for numerical data
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())])

    # Preprocessing for categorical data
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, ['Age', 'Work_Experience', 'Family_Size']),
            ('cat', categorical_transformer, ['Gender', 'Ever_Married', 'Graduated', 'Profession', 'Spending_Score', 'Var_1'])])

    # Define the XGBoost model
    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')  # Adjust XGBoost parameters as needed

    # Bundle preprocessing and modeling code in a pipeline
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('model', model)])

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

    # Preprocessing of training data, fit model 
    clf.fit(X_train, y_train)

    # Preprocessing of validation data, get predictions
    preds = clf.predict(X_test)

    # Evaluate the model
    report = classification_report(y_test, preds)
    return report

# The argparse and main block have been removed as requested
