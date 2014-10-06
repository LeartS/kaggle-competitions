import math
import argparse
from datetime import datetime
import numpy as np
from sklearn import cross_validation
from sklearn import linear_model
from sklearn import metrics

def load_data(path, **kwargs):
    return np.loadtxt(path, **kwargs)

def save_data(path, data, **kwargs):
    np.savetxt(path, data, **kwargs)

def hour_from_dt_string(dt_string):
    return datetime.strptime(dt_string, '%Y-%m-%d %H:%M:%S').hour

def cv(estimator, X, y):
    k_fold = cross_validation.KFold(n=len(train_dataset), n_folds=10,
                                    indices=True)
    a = 0.0
    for train_idx, test_idx in k_fold:
        r = estimator.fit(X[train_idx], y[train_idx]).predict(X[test_idx])
        r = np.where(r > 0, r, 0.01)
        s = math.sqrt(metrics.mean_squared_error(np.log(y[test_idx] + 1),
                                                 np.log(r + 1.0)))
        a += s
        print 'Score: {:.4f}'.format(s)
    print 'Average score: {:.4f}'.format(a/len(k_fold))

if __name__ == '__main__':
    # Command arguments
    parser = argparse.ArgumentParser(description='bike-sharing estimator')
    parser.add_argument('--cv', dest='cv', action='store_const', const=True,
                        default=False, help='Do cross validation')
    parser.add_argument('--no-test', dest='out', action='store_const',
                        const=False, default=True, help='No test dataset')
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

    # The interesting part
    estimator = linear_model.LinearRegression()
    if args.cv:
        cv(estimator, train_dataset[:,:-1], train_dataset[:,-1])
    if args.out:
        results = estimator.fit(
            train_dataset[:,:-1], train_dataset[:,-1]
        ).predict(test_dataset)
        results = np.where(results > 0, results, 0.01).astype(np.int)

        # Output
        save_data('data/out.csv', np.column_stack((out_column.T, results.T)),
                  delimiter=',', header='datetime,count', fmt=('%s', '%s'),
                  comments='')
