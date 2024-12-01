from sklearn.model_selection import train_test_split, RepeatedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import pandas as pd

# from sklearn.model_selection import RepeatedKFoldKFold
from sklearn.model_selection import cross_val_score

# making the assumption that this is training data
train_df = pd.read_csv('../../feature_data/train/featureInput.csv')
test_df = pd.read_csv('../../feature_data/test/featureInput.csv')

drop_cols = list(train_df.filter(regex='ChampionRole$').columns)

train_df = train_df.drop(drop_cols, axis=1)
test_df = test_df.drop(drop_cols, axis=1)
# if the win column is not combined with the original dataset, combine here


    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=i)
    # kf = RepeatedKFoldKFold(n_repeats=5, n_splits=10, random_state=i, shuffle=True)
    # okay, I will be working off of the assumption that the data has already been split
    # for training and testing
    # this shuffling probably is not that necessary either
    # train_df = train_data.sample(frac=1, random_state=i)
y_train = train_df.loc[:, 'bResult']
X_train = train_df.loc[:, train_df.columns != 'bResult']
y_test = test_df.loc[:, 'bResult']
X_test = test_df.loc[:, test_df.columns != 'bResult']
# given that this is for training, no prior probabilities will be provided
# when run on the test data, will need to include the priors from the training data
gnb = GaussianNB()
gnb.fit(X_train, y_train)
# model must be fit before creating priors
# actually not sure how necessary this is
priors = gnb.predict_proba(X_train)

y_predict = gnb.predict(X_test)
mean_accuracy = gnb.score(X_test, y_test)
accuracy = accuracy_score(y_test, y_predict, normalize=True)

def __init__(self):
    return gnb