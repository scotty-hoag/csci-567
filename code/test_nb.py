import time

import numpy as np
import pandas as pd
import xgbStack
import load_data

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve, cross_val_score

from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

gnb = GaussianNB()

def predict_shuffled_y(x_train, y_train, y_test, y_data_full_df):
    y_shuffled = y_data_full_df.sample(frac=1, random_state=42).reset_index(drop=True)
    y_train_shuffled = y_shuffled[y_train.index]
    y_test_shuffled = y_shuffled[y_test.index]

    gnb.fit(x_train, y_train_shuffled)
    y_pred_shuffled = gnb.predict(y_test)

    shuffled_accuracy = accuracy_score(y_test_shuffled, y_pred_shuffled)
    print(f"Accuracy on shuffled labels: {shuffled_accuracy:.2f}")

def perform_cross_validation(model, x_full_df, y_full_df):

    pipeline = Pipeline([
        ('scaler', StandardScaler()),       # Step 1: Scale the data
        ('classifier', model)               # Step 2: Apply the model
    ])
    
    scores = cross_val_score(pipeline, x_full_df, y_full_df, cv=10)
    print("Cross-validation scores:", scores)
    print("Mean cross-validation score:", scores.mean())

def test_using_import_df():
    modelInstance = xgbStack.xgbStack()

    match_df_train, match_df_test, y_train, y_test, x_data_full_df, y_data_full_df = load_data.load_matchdata_into_df("original")
    # x_train = modelInstance.perform_z_score(match_df_train)
    # x_test = modelInstance.perform_z_score(match_df_test)

    gnb = GaussianNB()
    gnb.fit(match_df_train, y_train)
    y_pred = gnb.predict(match_df_test)    

    x_combined = pd.concat([match_df_train, match_df_test])
    y_combined = pd.concat([y_train, y_test])

    perform_cross_validation(gnb, x_combined, y_combined)
    
    
def test_using_csv_import():
    modelInstance = xgbStack.xgbStack()
    match_df_train = modelInstance.import_data(True, bGenerateOutputFile=False)
    match_df_test = modelInstance.import_data(False, bGenerateOutputFile=False)
    
    x_train, y_train = modelInstance.extract_labels(match_df_train)
    x_test, y_test = modelInstance.extract_labels(match_df_test)    

    # x_train = modelInstance.perform_z_score(x_train)
    # x_test = modelInstance.perform_z_score(x_test)

    gnb = GaussianNB()
    gnb.fit(x_train, y_train)
    y_pred = gnb.predict(x_test)

    x_combined = pd.concat([x_train, x_test])
    y_combined = pd.concat([y_train, y_test])

    scaler = StandardScaler()
    scaler.fit(x_combined)
    x_combined = scaler.transform(x_combined)
    

    perform_cross_validation(gnb, x_combined, y_combined)

if __name__ == "__main__":

    test_using_csv_import()
    # test_using_import_df()
