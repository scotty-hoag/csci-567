# Import necessary libraries
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import xgbStack

# Assuming you already have your datasets and labels
# X_train, X_test, y_train, y_test

if __name__ == "__main__":
    modelInstance = xgbStack.xgbStack()
    match_df_training = modelInstance.import_data(True, bGenerateOutputFile=False)
    match_df_test = modelInstance.import_data(False, bGenerateOutputFile=False)
    
    x_train, y_train = modelInstance.extract_labels(match_df_training)
    x_test, y_test = modelInstance.extract_labels(match_df_test)

    x_train = modelInstance.perform_z_score(x_train)
    x_test = modelInstance.perform_z_score(x_test)

    # Initialize the MLPClassifier with some hyperparameters
    mlp = MLPClassifier(hidden_layer_sizes=(100,),  # 1 hidden layer with 100 neurons
                        activation='relu',          # Activation function for hidden layers
                        solver='adam',               # Optimizer
                        max_iter=200,                # Number of epochs
                        random_state=42)             # Seed for reproducibility

    # Train the model
    mlp.fit(x_train, y_train)

    # Predict on the test set
    y_pred = mlp.predict(x_test)

    # Calculate accuracy and display classification report
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
