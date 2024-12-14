import time
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score, RepeatedKFold, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import StackingClassifier #For XGB implementation
from scipy.stats import zscore

from sklearn.naive_bayes import GaussianNB, MultinomialNB
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

    def test_perform_cross_validation(self, pipeline, x_training, y_training, bPerformOOF=False):

        #The default parameters in the paper menvbghtion n_splits=10, n_repeats=5, but this configuration can take a long time to process.
        repeated_kfold = RepeatedKFold(n_splits=10, n_repeats=1, random_state=84)
        # repeated_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        if bPerformOOF:
            # Arrays to store OOF predictions and test indices
            oof_predictions = pd.Series(index=y_training, dtype=float)
            fold_metrics = []

            for train_idx, valid_idx in repeated_kfold.split(x_training):
                # Split the data into training and validation sets
                X_train, X_valid = x_training.iloc[train_idx], x_training.iloc[valid_idx]
                y_train, y_valid = y_training.iloc[train_idx], y_training.iloc[valid_idx]
                
                # Train the model on the training data
                pipeline.fit(X_train, y_train)
                
                # Predict on the validation set                
                # oof_predictions.iloc[valid_idx] = pipeline.predict(X_valid)
                valid_preds = pipeline.predict(X_valid)
                oof_predictions.iloc[valid_idx] = valid_preds

                # Evaluate the fold performance
                fold_auc = roc_auc_score(y_valid, valid_preds)
                fold_metrics.append(fold_auc)

            # Evaluate the performance of the OOF predictions
            oof_accuracy = accuracy_score(y_training, oof_predictions)
            print(f"Out-of-Fold Accuracy: {oof_accuracy:.4f}")

            # Compute overall OOF metric
            oof_auc = roc_auc_score(y_training, oof_predictions)
            print(f"Mean Fold AUC: {np.mean(fold_metrics):.4f}, Std: {np.std(fold_metrics):.4f}")
            print(f"Overall OOF AUC: {oof_auc:.4f}")            

        else:
            scores = cross_val_score(pipeline, x_training, y_training, cv=repeated_kfold, scoring='accuracy')
            
            print("Cross-validation scores:", scores)
            print("Mean cross-validation score:", scores.mean())
            print("Standard deviation of cross-validation scores:", scores.std())

            # scores = cross_val_score(pipeline, x_training, y_training, cv=repeated_kfold, scoring='accuracy')

            # print(f"Cross-validated AUC scores: {scores}")
            # print(f"Mean AUC: {np.mean(scores):.4f}, Std: {np.std(scores):.4f}")

            # # Fit the final model on the entire dataset
            # pipeline.fit(x_training, y_training)


            # # Make predictions on the full dataset (for demonstration purposes)
            # final_predictions = pipeline.predict_proba(x_training)[:, 1]
            # final_auc = roc_auc_score(y_training, final_predictions)
            # print(f"\nFinal Model AUC on the Full Dataset: {final_auc:.4f}")            


    def test_perform_cv_base_models(self, base_models, pipeline, x_dataset, y_dataset):
        # Number of folds
        n_splits = 5
        kf = RepeatedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        # Prepare storage for OOF predictions
        oof_predictions_base = np.zeros((len(y_dataset), len(base_models)))  # Base model OOF predictions

        # Generate OOF predictions for base models
        for model_idx, (model_name, pipeline) in enumerate(base_models):
            print(f"Generating OOF predictions for {model_name}...")
            for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y), 1):
                print(f"  Processing Fold {fold}...")
                
                # Split the data into training and validation sets
                X_train, X_valid = x_dataset.iloc[train_idx], x_dataset.iloc[valid_idx]
                y_train, y_valid = y_dataset.iloc[train_idx], y_dataset.iloc[valid_idx]
                
                # Train the base model (pipeline)
                pipeline.fit(X_train, y_train)
                
                # Predict probabilities on the validation set
                valid_preds = pipeline.predict_proba(X_valid)[:, 1]
                oof_predictions_base[valid_idx, model_idx] = valid_preds

        # Train XGBoost meta-model on OOF predictions
        print("Training meta-model with XGBoost...")
        pipeline.fit(oof_predictions_base, y_dataset)

        # Evaluate ensemble model (stacking) on OOF predictions
        final_oof_predictions = pipeline.predict_proba(oof_predictions_base)[:, 1]
        
        oof_auc = roc_auc_score(y, final_oof_predictions)
        print(f"\nFinal Stacked Model OOF AUC: {oof_auc:.4f}")
        

    def train_model(self):
        load_data.generate_temp_csv_data()

        match_df_training = self.import_data(True, bIsGeneratedSet=True, bGenerateOutputFile=False)
        match_df_test = self.import_data(False, bIsGeneratedSet=True, bGenerateOutputFile=False)

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
            learning_rate=1e-6, #Note that the paper mentions 1e-6, but this causes the model to underperform significantly.
            min_child_weight=1,
            gamma=0,
            subsample=0.15,
            colsample_bytree=1e-9
        )

        stacking_clf = StackingClassifier(
            estimators=base_models, 
            final_estimator=meta_model,
            passthrough=True 
        )

        pipeline = Pipeline([
            ('scaler', StandardScaler()),         # Z-Score normalization occurs here
            ('stacking', stacking_clf)            # Step 2: Stacking Classifier with meta-model
        ])        

        #Debug sequence
        self.test_filter_columns(x_train, x_test)
        self.test_perform_cross_validation(pipeline, x_train, y_train)
        
        pipeline.fit(x_train, y_train)
        y_pred = pipeline.predict(x_test)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")

    def train_model_df(self, dirDataset='original', fileName='match_info.csv', test_split=0.2, 
                       bFull_cross_val_dataset=False, bPerformOOF=False):
        x_train, x_test, y_train, y_test, x_combined_df, y_combined_df = load_data.load_matchdata_into_df(
            dirDataset, inputFileName=fileName, test_split=test_split)

        #Placeholder implementation - should be replaced with model objects returned from respective base model .py files.
        model_nb = GaussianNB()
        # model_nb = MultinomialNB()
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
            # learning_rate=1e-6, #Note that the paper mentions 1e-6, but this causes the model to underperform significantly.
            min_child_weight=1,
            gamma=0,
            subsample=0.15,
            colsample_bytree=1e-9
        )

        stacking_clf = StackingClassifier(
            estimators=base_models, 
            final_estimator=meta_model,
            passthrough=True 
        )

        pipeline = Pipeline([
            ('scaler', StandardScaler()),         # Z-Score normalization occurs here
            ('stacking', stacking_clf)            # Step 2: Stacking Classifier with meta-model
        ])        

        #Debug sequence
        self.test_filter_columns(x_train, x_test)

        if not bFull_cross_val_dataset:
            self.test_perform_cross_validation(pipeline, x_train, y_train)
        else:            
            self.test_perform_cross_validation(pipeline, x_combined_df, y_combined_df, bPerformOOF)

        accuracy = pipeline.fit(x_train, y_train).score(x_test, y_test)
        # y_pred = pipeline.predict(x_test)

        # # Evaluate the model
        # accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")

    def train_base_model(self, dirDataset='original', fileName='match_info.csv', test_split=0.2, 
                       bFull_cross_val_dataset=False, bPerformOOF=False):
        x_train, x_test, y_train, y_test, x_combined_df, y_combined_df = load_data.load_matchdata_into_df(
            dirDataset, inputFileName=fileName, test_split=test_split)

        model = GaussianNB()
        # model_nn = MLPClassifier(hidden_layer_sizes=(135,),  
        #                 activation='tanh',          # Activation function for hidden layers
        #                 learning_rate_init=3e-3,
        #                 max_iter=500,
        #                 batch_size=2275,
        #                 alpha=0.0001,
        #                 learning_rate='adaptive',
        #                 beta_1=0.74,
        #                 solver='adam')     


        pipeline = Pipeline([
            ('scaler', StandardScaler()),         # Z-Score normalization occurs here
            ('model', model)            # Step 2: Stacking Classifier with meta-model
        ]) 

        #Debug sequence
        self.test_filter_columns(x_train, x_test)

        if not bFull_cross_val_dataset:
            self.test_perform_cross_validation(pipeline, x_train, y_train)
        else:            
            self.test_perform_cross_validation(pipeline, x_combined_df, y_combined_df, bPerformOOF)

        accuracy = pipeline.fit(x_train, y_train).score(x_test, y_test)
        # y_pred = pipeline.predict(x_test)

        # # Evaluate the model
        # accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")        

    def returnModel(self):        
        #Placeholder implementation - should be replaced with model objects returned from respective base model .py files.
        model_nb = GaussianNB()
        # model_nb = MultinomialNB()
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

        return pipeline


if __name__ == "__main__":
    fileName = "match_info_combined.csv"
    folderName = "new"
    modelInstance = xgbStack()

    startTime = time.time()

    # modelInstance.train_model()
    # modelInstance.train_model_df(dirDataset=folderName, fileName=fileName, test_split=0.1, 
    #                              bFull_cross_val_dataset=True, bPerformOOF=False)

    modelInstance.train_base_model(dirDataset=folderName, fileName=fileName, test_split=0.1, 
                                 bFull_cross_val_dataset=True, bPerformOOF=False)

    endTime = time.time()

    elapsed_time = round(endTime - startTime, 3)
    print(f"Execution time: {elapsed_time}") 