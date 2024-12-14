import time
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score, RepeatedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.ensemble import StackingClassifier #For XGB implementation
from scipy.stats import zscore

from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import load_data
import neuralnetwork
import sklearn_bayes

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
    def import_data(self, bIsTrainingSet, bIsGeneratedSet=True, bGenerateOutputFile=False, bIncludeChampionRole_Feature=False):
        df_input = load_data.load_data_from_csv(bIsTrainingSet, bIsGeneratedSet=bIsGeneratedSet, bGenerateOutputFile=bGenerateOutputFile, 
                                                bIncludeChampionRole_Feature=bIncludeChampionRole_Feature)
        
        return df_input
    
    def extract_labels(self, x_input):
        """
            Extracts the bResult column from the dataframe and returns another dataframe with those labels.            
        """
        y_data = x_input['bResult']
        x_data = x_input.drop(columns='bResult')

        return x_data, y_data
    
    def combine_dataset(self, x_train, x_test, y_train, y_test):
        """
            Combine the dataset for performing operations such as cross-validation. Passed data
            must not be normalized.            
        """
        x_combined = pd.concat([x_train, x_test]).reset_index()
        y_combined = pd.concat([y_train, y_test]).reset_index()

        x_combined.drop(columns="index",inplace=True)
        y_combined.drop(columns="index",inplace=True)

        return x_combined, y_combined
    
    def test_filter_columns(self, x_train, x_test):
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

        if len(list_dropcol) > 0:
            x_train.drop(columns=list_dropcol, inplace=True)
            x_test.drop(columns=list_dropcol, inplace=True) 

    def test_perform_cross_validation(self, pipeline, x_training, y_training):

        #The default parameters in the paper mention n_splits=10, n_repeats=5, but this configuration can take a long time to process.
        repeated_kfold = RepeatedKFold(n_splits=10, n_repeats=1, random_state=42)

        scores = cross_val_score(pipeline, x_training, y_training, n_jobs=1, cv=repeated_kfold, scoring="accuracy")

        print("Cross-validation scores:", scores)
        print("Mean cross-validation score:", scores.mean())
        print("Standard deviation of cross-validation scores:", scores.std())

    def train_model(self):        
        load_data.generate_temp_csv_data()

        match_df_training = self.import_data(True, bIsGeneratedSet=True, bGenerateOutputFile=True)
        match_df_test = self.import_data(False, bIsGeneratedSet=True, bGenerateOutputFile=True)

        x_train, y_train = self.extract_labels(match_df_training)
        x_test, y_test = self.extract_labels(match_df_test)    

        #Placeholder implementation - should be replaced with model objects returned from respective base model .py files.
        # model_nb = GaussianNB()
        model_nb = sklearn_bayes.returnModel(x_train, x_test, y_train, y_test)
        model_nn = neuralnetwork.get_lol_nnet_model(train_model=False, in_training_type=1, in_random_seed=42)
        
        # model_nn = MLPClassifier(hidden_layer_sizes=(256,),  
        #                 activation='tanh',          # Activation function for hidden layers
        #                 learning_rate_init=3e-4,
        #                 max_iter=500,
        #                 batch_size=2275,
        #                 alpha=0.0001,
        #                 learning_rate='adaptive',
        #                 beta_1=0.9,
        #                 solver='adam')     

        base_models = [
            ('naive_bayes', model_nb),
            ('neural_net', model_nn)
        ]

        meta_model = XGBClassifier(
            n_estimators=14000,
            max_depth=2,
            # learning_rate=1e-6, #Note that the paper mentions 1e-6, but this causes the model to underperform significantly.
            min_child_weight=1,
            gamma=0,
            subsample=0.15,
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

        #Debug sequence
        self.test_filter_columns(x_train, x_test)

        self.test_perform_cross_validation(pipeline, x_train, y_train)
        
        # pipeline.fit(x_train, y_train)
        y_pred = pipeline.predict(x_test)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")

    def train_model_df(self):
        """
            This function is for testing purposes only. Uses the pandas implementation to dynamically
            generate the test/train datasets instead of loading the CSV files.

        """
        x_train, x_test, y_train, y_test, x_combined_df, y_combined_df = load_data.load_matchdata_into_df("original")

        # x_train, y_train = self.extract_labels(match_df_training)
        # x_test, y_test = self.extract_labels(match_df_test)    

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
            booster='gbtree',
            n_estimators=14000,
            max_depth=2,
            learning_rate=1e-3, #Note that the paper mentions 1e-6, but this causes the model to underperform significantly.
            min_child_weight=1,
            gamma=0,
            subsample=0.15,
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

        # pipeline.fit(x_train, y_train)
        # y_pred = pipeline.predict(x_test)

        # Evaluate the model
        # accuracy = accuracy_score(y_test, y_pred)
        # print(f"Accuracy: {accuracy:.4f}")
        
        self.test_perform_cross_validation(pipeline, x_train, y_train)
        
        pipeline.fit(x_train, y_train)
        y_pred = pipeline.predict(x_test)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    modelInstance = xgbStack()

    startTime = time.time()

    modelInstance.train_model()
    # modelInstance.train_model_df()

    endTime = time.time()

    elapsed_time = round(endTime - startTime, 3)
    print(f"Execution time: {elapsed_time}") 