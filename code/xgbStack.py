import time
import numpy as np
import pandas as pd

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import StackingClassifier #For XGB implementation
from scipy.stats import zscore

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
    
    def perform_z_score(self, x_input):
        return x_input.apply(zscore)

    def parallelExecute_LearningModels(self):
        """
            Execute both the naive bayes model and NN concurrently.

        """
        # with ThreadPoolExecutor() as executor:
        #     pass

            # future_1 = executor.submit(train_and_predict, model_1, X_train, y_train, X_test)
            # future_2 = executor.submit(train_and_predict, model_2, X_train, y_train, X_test)
        
            # Retrieve the predictions
            # predictions_1 = future_1.result()
            # predictions_2 = future_2.result()
        

        #Return a tuple containing the generated dataframe predictions for each base model.

        
    def execute_metaModel(self, tuple_df_baseModelPredictions):
        #Refer to implementation described at: https://scikit-learn.org/dev/modules/generated/sklearn.ensemble.StackingClassifier.html 
        
        #xgb_model = StackingClassifier()

        pass
        
    def train_model(self):
        matchData_df = self.import_data()
        self.split_data()
        self.parallelExecute_LearningModels()
        self.execute_metaModel()

        #Return dataframe of final predictions.


if __name__ == "__main__":
    modelInstance = xgbStack()
    match_df_training = modelInstance.import_data(True, bGenerateOutputFile=False)
    match_df_test = modelInstance.import_data(False, bGenerateOutputFile=False)
    
    x_train, y_train = modelInstance.extract_labels(match_df_training)
    x_test, y_test = modelInstance.extract_labels(match_df_test)

    x_train = modelInstance.perform_z_score(x_train)
    x_test = modelInstance.perform_z_score(x_test)