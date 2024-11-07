import time

import numpy as np
import pandas as pd
import xgbStack
import load_data

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

gnb = GaussianNB()

if __name__ == "__main__":
    modelInstance = xgbStack.xgbStack()
    # match_df_training = modelInstance.import_data(True, bGenerateOutputFile=False)
    # match_df_test = modelInstance.import_data(False, bGenerateOutputFile=False)
    
    # x_train, y_train = modelInstance.extract_labels(match_df_training)
    # x_test, y_test = modelInstance.extract_labels(match_df_test)

    match_df_train, match_df_test, y_train, y_test = load_data.load_matchdata_into_df("original")
    
    x_train = modelInstance.perform_z_score(match_df_train)
    x_test = modelInstance.perform_z_score(match_df_test)

    gnb = GaussianNB()
    gnb.fit(x_train, y_train)
    y_pred = gnb.predict(x_test)

    # Evaluate the model's performance
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    train_sizes, train_scores, test_scores = learning_curve(gnb, x_train, y_train)

    # Print the results
    print("Accuracy:", accuracy)
    print("Confusion Matrix:\n", conf_matrix)
    print("Classification Report:\n", class_report)

