# Import necessary libraries
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import xgbStack
import pandas as pd

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout
# from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import learning_curve, cross_val_score

# Assuming you already have your datasets and labels
# X_train, X_test, y_train, y_test

def perform_cross_validation(model, x_full_df, y_full_df):

    pipeline = Pipeline([
        ('scaler', StandardScaler()),       # Step 1: Scale the data
        ('classifier', model)               # Step 2: Apply the model
    ])
    
    scores = cross_val_score(pipeline, x_full_df, y_full_df, cv=10)
    print("Cross-validation scores:", scores)
    print("Mean cross-validation score:", scores.mean())

def keras_implementation():    
    modelInstance = xgbStack.xgbStack()
    match_df_training = modelInstance.import_data(True, bGenerateOutputFile=False)
    match_df_test = modelInstance.import_data(False, bGenerateOutputFile=False)
    
    x_train, y_train = modelInstance.extract_labels(match_df_training)
    x_test, y_test = modelInstance.extract_labels(match_df_test)

    

    # Initialize the MLPClassifier with some hyperparameters
    mlp = MLPClassifier(hidden_layer_sizes=(256,),  
                        activation='tanh',          # Activation function for hidden layers
                        learning_rate_init=3e-4,
                        max_iter=500,
                        batch_size=2275,
                        alpha=0.0001,
                        learning_rate='adaptive',
                        beta_1=0.9)
                        # solver='adam',               # Optimizer                        
                        #random_state=42)             # Seed for reproducibility

    # Train the model
    mlp.fit(x_train, y_train)

    # Predict on the test set
    y_pred = mlp.predict(x_test)

    # Calculate accuracy and display classification report
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    x_combined = pd.concat([x_train, x_test])
    y_combined = pd.concat([y_train, y_test])

    perform_cross_validation(mlp, x_combined, y_combined)

def sklearn_implementation():
    modelInstance = xgbStack.xgbStack()
    match_df_training = modelInstance.import_data(True, bGenerateOutputFile=False)
    match_df_test = modelInstance.import_data(False, bGenerateOutputFile=False)
    
    x_train, y_train = modelInstance.extract_labels(match_df_training)
    x_test, y_test = modelInstance.extract_labels(match_df_test)

    # Initialize the MLPClassifier with some hyperparameters
    mlp = MLPClassifier(hidden_layer_sizes=(256,),  
                        activation='tanh',          # Activation function for hidden layers
                        learning_rate_init=3e-4,
                        max_iter=500,
                        batch_size=2275,
                        alpha=0.0001,
                        learning_rate='adaptive',
                        beta_1=0.9,
                        solver='adam')            # Optimizer                        
                        #random_state=42)             # Seed for reproducibility

    # Train the model
    mlp.fit(x_train, y_train)

    # Predict on the test set
    y_pred = mlp.predict(x_test)

    # Calculate accuracy and display classification report
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    x_combined = pd.concat([x_train, x_test])
    y_combined = pd.concat([y_train, y_test])

    perform_cross_validation(mlp, x_combined, y_combined)

if __name__ == "__main__":
    sklearn_implementation()
