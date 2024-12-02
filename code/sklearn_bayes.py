from sklearn.model_selection import RepeatedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

# from sklearn.model_selection import RepeatedKFoldKFold
from sklearn.model_selection import cross_val_score

# loading the datasets
train_df = pd.read_csv('../../feature_data/train/featureInput.csv')
test_df = pd.read_csv('../../feature_data/test/featureInput.csv')

# dropping the feature five columns just in case
drop_cols = list(train_df.filter(regex='ChampionRole$').columns)

train_df = train_df.drop(drop_cols, axis=1)
test_df = test_df.drop(drop_cols, axis=1)

# splitting the data into X and y sets
y_train = train_df.loc[:, 'bResult']
X_train = train_df.loc[:, train_df.columns != 'bResult']
y_test = test_df.loc[:, 'bResult']
X_test = test_df.loc[:, test_df.columns != 'bResult']

# for the sake of following the research paper, will be implementing
# a repeated ten-fold cross validation to find the prior probabilities
rkf = RepeatedKFold(n_repeats=5, n_splits=10, random_state=3)
best_probs = {'prob_blue': 0, 'prob_red': 0}
best_accuracy = 0
for train_index, test_index in rkf.split(X_train):
    # print(f'    Train: index={train_index}')
    # print(f'    Test: index={test_index}')
    X_fold_train, X_fold_test = X_train.iloc[train_index], X_train.iloc[test_index]
    y_fold_train, y_fold_test = y_train.iloc[train_index], y_train.iloc[test_index]
    prob_blue = len([x for x in list(y_fold_train) if x == 1]) / len(list(y_fold_train))
    prob_red = len([x for x in list(y_fold_train) if x == 0]) / len(list(y_fold_train))
    model = GaussianNB()
    model.fit(X_fold_train, y_fold_train)
    fold_prediction = model.predict(X_fold_test)
    accuracy = accuracy_score(y_fold_test, fold_prediction)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_probs['prob_blue'] = prob_blue
        best_probs['prob_red'] = prob_red

# fitting the model with the probabilities associated with the highest probability
# like the paper mentioned
gnb = GaussianNB(priors=[best_probs['prob_red'], best_probs['prob_blue']])
gnb.fit(X_train, y_train)
# model must be fit before creating priors
# actually not sure how necessary this is
priors = gnb.predict_proba(X_train)

y_predict = gnb.predict(X_test)
mean_accuracy = gnb.score(X_test, y_test)
accuracy = accuracy_score(y_test, y_predict, normalize=True)

def __init__(self):
    return gnb