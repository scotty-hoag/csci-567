import tensorflow as tf
import pandas as pd
import matplotlib as plt
from pathlib import Path

HPARAMS_DEFAULTS = 0
HPARAMS_PAPER = 1
HPARAMS_CUSTOM = 2
training_type = HPARAMS_CUSTOM

OP_RMSPROP = 0
OP_SGD = 1
op_type = OP_RMSPROP

if (training_type == HPARAMS_DEFAULTS):
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

elif (training_type == HPARAMS_PAPER):
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

def compile_neural_network():
    output_layer_activation_function = "sigmoid"
    tf.keras.utils.set_random_seed(42)

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
        
    return model