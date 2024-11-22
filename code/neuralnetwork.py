import tensorflow as tf
import pandas as pd
import matplotlib as plt
from pathlib import Path

import os; 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

HPARAMS_DEFAULTS = 0
HPARAMS_PAPER = 1
HPARAMS_CUSTOM = 2

OP_RMSPROP = 0
OP_SGD = 1


glob_training_path_without_champions = "./../feature_data/train/featureInput_noChampionRole_ZSCORED.csv"
glob_training_path_with_champions = "./../feature_data/train/featureInput_ZSCORED.csv"
glob_training_path_default = glob_training_path_without_champions

glob_testing_path_without_champions = "./../feature_data/test/featureInput_noChampionRole_ZSCORED.csv"
glob_testing_path_with_champions = "./../feature_data/test/featureInput_ZSCORED.csv"
glob_testing_path_default = glob_testing_path_without_champions


def get_lol_nnet_model(train_model=True, in_training_type=HPARAMS_CUSTOM, in_random_seed=42, in_training_data_path=glob_training_path_default, in_testing_data_path=glob_testing_path_default, print_debug=False):

    training_type = in_training_type
    op_type = OP_RMSPROP

    output_layer_activation_function = "sigmoid"
    tf.keras.utils.set_random_seed(in_random_seed)

    num_hidden_layers = 1
    num_hidden_layer_neurons = 32   # X
    dropout_rate = 0.4              # X
    batch_size = 32                 # X 
    optimizer_learning_rate = 0.01  # X  
    gradient_decay_rho = 0.9
    learning_rate_decay = 0.2       # X
    hidden_layer_activation_function = "relu"
    #-----
    op_type = OP_RMSPROP
    training_epochs = 30

    if (training_type == HPARAMS_PAPER):
        num_hidden_layers = 1
        num_hidden_layer_neurons = 256  # X
        dropout_rate = 0.4              # X
        batch_size = 2275               # X 
        optimizer_learning_rate = 3e-4  # X // 3e-4, or 3*(10^-4) = 0.0003
        gradient_decay_rho = 0.9
        learning_rate_decay = 0.2       # X
        hidden_layer_activation_function = "tanh"
        #-----
        op_type = OP_RMSPROP
        training_epochs = 30
        
    elif (training_type == HPARAMS_CUSTOM):
        num_hidden_layers = 1
        num_hidden_layer_neurons = 135     # X
        dropout_rate = 0.2596709650832801  # X
        batch_size = 2275                  # X 
        optimizer_learning_rate = 0.0029044390072926946   # X // 3e-4, or 3*(10^-4) = 0.0003
        gradient_decay_rho = 0.7437864402038941
        learning_rate_decay = 0.39953909944182836         # X
        hidden_layer_activation_function = "tanh"
        #-----
        op_type = OP_RMSPROP
        training_epochs = 30

    # Create the model
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=[28]))
    model.add(tf.keras.layers.Dropout(rate=dropout_rate))
    for _ in range(num_hidden_layers):
        model.add(tf.keras.layers.Dense(num_hidden_layer_neurons, activation=hidden_layer_activation_function))
        model.add(tf.keras.layers.Dropout(rate=dropout_rate))
    model.add(tf.keras.layers.Dense(1, activation=output_layer_activation_function))

    if op_type == OP_RMSPROP:
        model.compile(
            loss="binary_crossentropy",
            optimizer=tf.keras.optimizers.RMSprop(learning_rate=optimizer_learning_rate, rho=gradient_decay_rho),
            metrics=["accuracy"])
    else:
        model.compile(
            loss="binary_crossentropy",
            optimizer=tf.keras.optimizers.SGD(learning_rate=optimizer_learning_rate, decay=learning_rate_decay),
            metrics=["accuracy"])
    
    if not train_model:
        return model

    # Load the training data.
    train_path = Path(in_training_data_path)
    test_path = Path(in_testing_data_path)

    #names = ["Match Result", "Blue Top Player", "Blue Jungle Player", "Blue Middle Player", "Blue ADC Player", "Blue Support Player", "Red Top Player", "Red Jungle Player", "Red Middle Player", "Red ADC Player", "Red Support Player", "Blue Top Champion", "Blue Jungle Champion", "Blue Middle Champion", "Blue ADC Champion", "Blue Support Champion", "Red Top Champion", "Red Jungle Champion", "Red Middle Champion", "Red ADC Champion", "Red Support Champion", "Blue Player Cooperation", "Red Player Cooperation", "Blue vs Red Player", "Blue Champion Cooperation", "Red Champion Cooperation", "Blue vs Red Champion", "Blue Team Win Rate When Blue", "Red Team Win Rate When Red"]
    names = ["bResult", "btPlayerRole", "bjPlayerRole", "bmPlayerRole", "baPlayerRole", "bsPlayerRole", "rtPlayerRole", "rjPlayerRole", "rmPlayerRole", "raPlayerRole", "rsPlayerRole", "btPlayerChampion", "bjPlayerChampion", "bmPlayerChampion", "baPlayerChampion", "bsPlayerChampion", "rtPlayerChampion", "rjPlayerChampion", "rmPlayerChampion", "raPlayerChampion", "rsPlayerChampion", "bCoopPlayer", "rCoopPlayer", "vsPlayer", "bCoopChampion", "rCoopChampion", "vsChampion", "bTeamColor", "rTeamColor"]

    # Load our pre-processed training and test sets. 
    # The skiprows value just removes the first row of the CSV which are labels instead of data.
    train_data = (pd.read_csv(train_path, names=names, skiprows=1))
    test_data  = (pd.read_csv( test_path, names=names, skiprows=1))

    y_train = train_data["bResult"]
    y_test  =  test_data["bResult"]

    # Remove the match result column from the data.
    X_train = train_data.drop("bResult", axis=1)
    X_test  =  test_data.drop("bResult", axis=1)
    
    # Train the model.
    history = model.fit(
        X_train, y_train, 
        validation_data=(X_test, y_test),
        epochs=training_epochs,
        batch_size=batch_size,
        verbose=(1 if print_debug else 0)
    )
    return model


if __name__ == "__main__":
    import sys

    train_path = glob_training_path_default
    test_path = glob_testing_path_default

    args = sys.argv[1:]
    for i in range(len(args)):
        if (args[i] == "-train"):
            i += 1
            train_path = args[i]
        if (args[i] == "-test"):
            i += 1
            test_path = args[i]

    get_lol_nnet_model(train_model=True, in_training_type=HPARAMS_CUSTOM, in_random_seed=42, in_training_data_path=train_path, in_testing_data_path=test_path, print_debug=True)

    exit(0)