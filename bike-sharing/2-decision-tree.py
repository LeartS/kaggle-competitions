import math
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn import tree
from sklearn import metrics

# plt.style.use('ggplot')

def preprocessing(X, y):
    X['weekday'] = X.index.weekday
    X['hour'] = X.index.hour
    X['year'] = X.index.year
    return X, y

def scoring(y_real, y_predicted):
    y_real = y_real.round().astype(np.int)
    y_predicted = np.where(y_predicted > 0, y_predicted, 0).round().astype(np.int)
    return math.sqrt(
        metrics.mean_squared_error(np.log(y_predicted + 1), np.log(y_real + 1))
    )

def cv(estimator, X, y, day_split_first=8, day_split_last=13):
    day_split = (a for a in
                 [np.where(X.index.day < i) + np.where(X.index.day >= i)
                  for i in xrange(day_split_first, day_split_last+1)])
    scores = cross_validation.cross_val_score(estimator, X, y, cv=day_split,
                                              score_func=scoring)
    print scores.round(3)
    print 'Avg: {1:.3f} | std: {2:.3f} | min: {3:.3f} | max: {6:.3f}'.format(
        *pd.Series(scores).describe()[:]
    )

if __name__ == '__main__':
    # Command arguments
    parser = argparse.ArgumentParser(description='bike-sharing estimator')
    parser.add_argument('--no-cv', dest='cv', action='store_false',
                        default=True, help='Skip cross validation')
    parser.add_argument('--no-test', dest='test', action='store_false',
                        default=True, help='Skip test dataset')
    parser.add_argument('-f', dest='first', type=int,
                        default=8, help='First starting day of test split')
    parser.add_argument('-l', dest='last', type=int,
                        default=13, help='Last starting day of test split')
    parser.add_argument('-d', '--max-depth', dest='depth', type=int,
                        default=10, help='Max depth of decision tree')
    args = parser.parse_args()

    # Input
    common_input_options = {'delimiter': ',', 'index_col': 0, 'parse_dates': True}
    train_dataset = pd.read_csv('data/train.csv', **common_input_options)
    test_dataset = pd.read_csv('data/test.csv', usecols=(0,1,2,3,4,5,6,7,8),
                               **common_input_options)

    # Data preprocessing
    X_train, y_train = preprocessing(train_dataset.iloc[:,:-3],
                                     train_dataset.iloc[:,-1])
    X_test, y_test = preprocessing(test_dataset, None)

    # The interesting part
    estimator = tree.DecisionTreeRegressor(max_depth=args.depth)
    if args.cv:
        cv(estimator, X_train, y_train, args.first, args.last)
    if args.test:
        results = estimator.fit(X_train, y_train).predict(X_test)
        results = np.where(results > 0, results, 0).round().astype(np.int)

        # Output
        output_dataframe = pd.DataFrame({'datetime': test_dataset.index})
        output_dataframe['count'] = results
        output_dataframe.to_csv('data/out.csv', index=False)
