import time
import numpy as np
import pandas as pd

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import StackingClassifier #For XGB implementation
from scipy.stats import zscore

from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import load_data

class xgbStack:
    def __init__(self, hpNaiveBayes=None, hpNeuralNetwork=None, hpXgb=None):
        """
            Input:
                hpNaiveBayes - dict: Dictionary whose keys represent the hyperparameters to 
                    set for naive bayes.
                hpNeuralNetwork - dict: Dictionary whose keys represent the hyperparameters
                    to set to for the neural network. 
                hpXbg - dict: Dictionary whose keys represent the hyperparameters to set
                    for extreme gradient boost.

        """
        pass

    def train_and_predict(self, model, X_train, y_train, X_test):
        model.fit(X_train, y_train)
        return model.predict(X_test)
    
    #Import the data, either by processing all the feature csv files along with the matchdata.csv file or 
    #inputting a precompiled input file.
    def import_data(self, bIsTrainingSet, bGenerateOutputFile=False, bIncludeChampionRole_Feature=False):
        df_input = load_data.load_data_from_csv(bIsTrainingSet, bGenerateOutputFile=bGenerateOutputFile, 
                                                bIncludeChampionRole_Feature=bIncludeChampionRole_Feature)
        
        return df_input
    
    def extract_labels(self, x_input):
        """
            Extracts the bResult column from the dataframe and returns another dataframe with those labels.            
        """
        y_data = x_input['bResult']
        x_data = x_input.drop(columns='bResult')

        return x_data, y_data

        
    def execute_metaModel(self, tuple_df_baseModelPredictions):
        #Refer to implementation described at: https://scikit-learn.org/dev/modules/generated/sklearn.ensemble.StackingClassifier.html 
        
        #xgb_model = StackingClassifier()

        pass
        
    def train_model(self):
        match_df_training = self.import_data(True, bGenerateOutputFile=False)
        match_df_test = self.import_data(False, bGenerateOutputFile=False)

        x_train, y_train = self.extract_labels(match_df_training)
        x_test, y_test = self.extract_labels(match_df_test)      

        model_nb = GaussianNB()
        model_nn = MLPClassifier

        pipeline = Pipeline([
            ('scaler', StandardScaler()),       # Step 1: Scale the data
            ('classifier', model_nb)            # Step 2: Apply the model

        ])          


        pass
        #Return dataframe of final predictions.


if __name__ == "__main__":
    modelInstance = xgbStack()
    modelInstance.train_model()


    # match_df_training = modelInstance.import_data(True, bGenerateOutputFile=False)
    # match_df_test = modelInstance.import_data(False, bGenerateOutputFile=False)

    # x_train, y_train = modelInstance.extract_labels(match_df_training)
    # x_test, y_test = modelInstance.extract_labels(match_df_test)

    # x_train = modelInstance.perform_z_score(x_train)
    # x_test = modelInstance.perform_z_score(x_test)