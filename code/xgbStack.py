import time

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from concurrent.futures import ThreadPoolExecutor

import load_data

class xgbStack:
    def __init__(self, hpNaiveBayes=None, hpNeuralNetork=None, hpXgb=None):
        pass

    def train_and_predict(self, model, X_train, y_train, X_test):
        model.fit(X_train, y_train)
        return model.predict(X_test)
    
    #Import the data, either by processing all the feature csv files along with the matchdata.csv file or 
    #inputting a precompiled input file.
    def import_data(self):
        pass
    
    #Split the data into training and test sets.
    def split_data(self, dataFrame):
        pass

    def parallelExecute_LearningModels(self):
        with ThreadPoolExecutor() as executor:
            pass

            # future_1 = executor.submit(train_and_predict, model_1, X_train, y_train, X_test)
            # future_2 = executor.submit(train_and_predict, model_2, X_train, y_train, X_test)
        
            # Retrieve the predictions
            # predictions_1 = future_1.result()
            # predictions_2 = future_2.result()
        
    def execute_metaModel(self):
        pass
        
    def train_model(self):

        self.import_data() #Should return a dataframe containing the labels mentioned in the research paper
        self.split_data()
        #self.parallelExecute_LearningModels()
        self.execute_metaModel()
        
        

if __name__ == "__main__":
    pass    