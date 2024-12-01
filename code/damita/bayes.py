import csv
import pandas as pd
import load_data
import numpy as np
import math
import random

# will need to change the file path, but for now this is fine
# load data into df somehow with the new features
# load the win column from the original dataset for the training data
# need to find the probabilities for P(blue) and P(red) for the final prediction

# split the data into train and test splits
# when they use seeds, I am assuming that the seeds
# are different for each repetition
# but it is a little difficult to tell from their explanation

num_data = df.count()


# need to separate training data by red and blue based on which
# team won and then find the summary statistics
def stat_calc(df):
    cols = {}
    for col_name in red_df.columns:
        col_data = list(df.loc[:, col_name])
        col_mean = np.mean(col_data)
        col_std = np.std(col_data)
        cols[col_name] = (col_mean, col_std, len(col_data))
    return cols

# dictionary with the summary statistics for each column for class red
red_cols = stat_calc(red_df)

# dictionary with the summary statistics for each column for class blue
blue_cols = stat_calc(blue_df)

# probability of the winner being red
# based on the training data
# P(red)
p_red = red_df.count() / df.count()

# probability of the winner being blue
# based on the training data
# P(blue)
p_blue = blue_df.count() / df.count()

def gaussian_prob(x, mean, std):
    e = math.exp(-(((x - mean) / std)**2) / 2)
    result = (1 / (math.sqrt(2 * math.pi) * std)) * e
    return result

# loring mentioned some values were removed, so the indices will not exactly match
# for my usage here, will want to create new column with the actual value for the winner
# to make processing easier

def predict(feature_probs):
    prediction_red = p_red
    prediction_blue = p_blue
    for x in feature_probs:
        prediction_red *= feature_probs[x][0]
        prediction_blue *= feature_probs[x][1]
    return prediction_red, prediction_blue



# matrix of P(x|color) values for each row in df to be used
# to predict unknown data (after validation of course)
def bayes(df):
    prob_blue = []
    prob_red = []
    win = []
    model_parameters = []

    for i in range(df.count()):
        #col_probs = {}
        # row_probs is P(x|color) for each column for that row of data
        row_probs = []
        for col in df.columns:
            red_mean = red_cols[col][0]
            red_std = red_cols[col][1]
            # don't know if this is the correct way to select
            data = df.loc[i, col]
            col_red = gaussian_prob(data, red_mean, red_std)
            blue_mean = blue_cols[col][0]
            blue_std = blue_cols[col][1]
            col_blue = gaussian_prob(data, blue_mean, blue_std)
            #col_probs[col] = (col_red, col_blue)
            row_probs.append((col_red, col_blue))
        model_parameters.append(row_probs)
        # predictions below must be validated for correctness
        predictions = predict(row_probs)
        prob_red.append(predictions[0])
        prob_blue.append(predictions[1])
        if predictions[0] > predictions[1]:
            win.append(False)
        elif predictions[0] < predictions[1]:
            win.append(True)
        else:
            win.append(0)
    return prob_blue, prob_red, win

# this should be the end of training
# not sure how to validate model yet
# when using the test data, the values found from gaussian prob will be used to predict new data
# so, I suppose I do need to edit the above to store that data in a matrix or separate dataframe
# i also need to figure out where the other parameters go
# I know that the laplace parameter is just 0
# i am not sure about the other one yet

# testing model on test_df

# might want to tack on the predictions to the train and test dfs
train_result = bayes(train_df)
test_result = bayes(test_df)

# okay, i think that my initial approach was wrong.
# so the model is run 5 times with 9 folds for training and 1 for testing
# then after the 5 repetitions, i am not sure what happens next
# i assume that i must take the average of the model's accuracy
# cv_split will be moved to the beginning of the program
fold_size = 10
repetitions = 5

def cv_split(df, folds):
    sample_size = df.count() / folds
    for i in range(folds):
        df.sample(n=sample_size, random_state=i)