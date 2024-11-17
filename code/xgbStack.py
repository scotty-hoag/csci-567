import time
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import StackingClassifier #For XGB implementation
from scipy.stats import zscore

from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import load_data
import neural_network

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
    
    def combine_dataset(x_train, x_test, y_train, y_test):
        """
            Combine the dataset for performing operations such as cross-validation. Passed data
            must not be normalized.            
        """
        x_combined = pd.concat([x_train, x_test])
        y_combined = pd.concat([y_train, y_test])

        return x_combined, y_combined
    
    def test_filter_columns(x_train, x_test):
        """
            For testing purposes only. Drops columns from the dataset
        """
        #Note: If an entry is NOT commented out, then the column will be dropped.
        list_dropcol = [    
            # 'btPlayerRole', 
            # 'bjPlayerRole', 
            # 'bmPlayerRole',       
            # 'baPlayerRole', 
            # 'bsPlayerRole',     
            # 'rtPlayerRole', 
            # 'rjPlayerRole',
            # 'rmPlayerRole', 
            # 'raPlayerRole', 
            # 'rsPlayerRole', 
            # 'btPlayerChampion',
            # 'bjPlayerChampion', 
            # 'bmPlayerChampion', 
            # 'baPlayerChampion',
            # 'bsPlayerChampion', 
            # 'rtPlayerChampion', 
            # 'rjPlayerChampion',
            # 'rmPlayerChampion', 
            # 'raPlayerChampion', 
            # 'rsPlayerChampion',
            # 'bCoopPlayer', 
            # 'rCoopPlayer', 
            # 'vsPlayer', 
            # 'bCoopChampion',
            # 'rCoopChampion', 
            # 'vsChampion', 
            # 'bTeamColor', 
            # 'rTeamColor'
        ] 

        x_train.drop(columns=list_dropcol, inplace=True)
        x_test.drop(columns=list_dropcol, inplace=True)        

    def train_model(self):
        match_df_training = self.import_data(True, bGenerateOutputFile=False)
        match_df_test = self.import_data(False, bGenerateOutputFile=False)

        x_train, y_train = self.extract_labels(match_df_training)
        x_test, y_test = self.extract_labels(match_df_test)    

        #Placeholder implementation - should be replaced with model objects returned from respective base model .py files.
        model_nb = GaussianNB()
        model_nn = MLPClassifier(hidden_layer_sizes=(256,),  
                        activation='tanh',          # Activation function for hidden layers
                        learning_rate_init=3e-4,
                        max_iter=500,
                        batch_size=2275,
                        alpha=0.0001,
                        learning_rate='adaptive',
                        beta_1=0.9,
                        solver='adam')     

        base_models = [
            ('naive_bayes', model_nb),
            ('neural_net', model_nn)
        ]

        meta_model = XGBClassifier(
            n_estimators=14000,
            max_depth=2,
            learning_rate=1e-2, #Note that the paper mentions 1e-6, but this causes the model to underperform significantly.
            min_child_weight=1,
            gamma=0,
            subsample=1e-1,
            colsample_bytree=1e-9
        )

        stacking_clf = StackingClassifier(
            estimators=base_models, 
            final_estimator=meta_model       
        )

        pipeline = Pipeline([
            ('scaler', StandardScaler()),         # Z-Score normalization occurs here
            ('stacking', stacking_clf)            # Step 2: Stacking Classifier with meta-model
        ])        

        pipeline.fit(x_train, y_train)
        y_pred = pipeline.predict(x_test)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        
        # cv_scores = cross_val_score(pipeline, x_combined, y_combined, cv=5)

if __name__ == "__main__":
    modelInstance = xgbStack()
    modelInstance.train_model()