import math
import argparse
from datetime import datetime
import numpy as np
from sklearn import cross_validation
from sklearn import tree
from sklearn import metrics

def load_data(path, **kwargs):
    return np.loadtxt(path, **kwargs)

def save_data(path, data, **kwargs):
    np.savetxt(path, data, **kwargs)

def hour_from_dt_string(dt_string):
    return datetime.strptime(dt_string, '%Y-%m-%d %H:%M:%S').hour

def preprocessing(X, y):
    is_seasons = np.empty((X.shape[0], 4))
    return X, y

def cv(estimator, X, y, n_folds=5, print_single=True):
    k_fold = cross_validation.KFold(n=len(train_dataset), n_folds=n_folds,
                                    indices=True)
    scores = []
    for train_idx, test_idx in k_fold:
        r = estimator.fit(X[train_idx], y[train_idx]).predict(X[test_idx])
        r = np.where(r > 0, r, 0).astype(np.int)
        s = math.sqrt(metrics.mean_squared_error(np.log(y[test_idx] + 1),
                                                 np.log(r + 1.0)))
        scores.append(s)
        if print_single:
            print 'Score: {:.3f}'.format(s)
    print 'Average score: {:.3f} - std: {:.4f}'.format(np.mean(scores),
                                                       np.std(scores))

def loss_func(y_real, y_predicted):
    return math.sqrt(metrics.mean_squared_error(np.log(y_real + 1), np.log(y_predicted + 1)))

if __name__ == '__main__':
    # Command arguments
    parser = argparse.ArgumentParser(description='bike-sharing estimator')
    parser.add_argument('--no-cv', dest='cv', action='store_false',
                        default=True, help='Skip cross validation')
    parser.add_argument('--no-test', dest='test', action='store_false',
                        default=True, help='Skip test dataset')
    parser.add_argument('-k', dest='n_fold', type=int,
                        default=5, help='Number of cv folds')
    parser.add_argument('--no-single', dest='single', action='store_false',
                        default=True, help='Print only average cv score')
    parser.add_argument('-d', '--max-depth', dest='depth', type=int,
                        default=10, help='Max depth of decision tree')
    args = parser.parse_args()

    # Input
    common_input_options = {'delimiter': ',', 'skiprows': 1,
                            'converters': {0: hour_from_dt_string} }
    train_dataset = load_data('data/train.csv', usecols=(0,1,2,3,4,5,6,7,8,11),
                              **common_input_options)
    test_dataset = load_data('data/test.csv', usecols=(0,1,2,3,4,5,6,7,8),
                             **common_input_options)
    common_input_options['converters'] = {}
    out_column = load_data('data/test.csv', usecols=(0,), dtype=str,
                           **common_input_options)

    # Data preprocessing
    X_train, y_train = preprocessing(train_dataset[:,:-1], train_dataset[:,-1])
    X_test, y_test = preprocessing(test_dataset, None)

    # The interesting part
    estimator = tree.DecisionTreeRegressor(max_depth=args.depth)
    if args.cv:
        cv(estimator, X_train, y_train, args.n_fold, args.single)
    if args.test:
        results = estimator.fit(X_train, y_train).predict(X_test)
        results = np.where(results > 0, results, 0.01).astype(np.int)

        # Output
        save_data('data/out.csv', np.column_stack((out_column.T, results.T)),
                  delimiter=',', header='datetime,count', fmt=('%s', '%s'),
                  comments='')
