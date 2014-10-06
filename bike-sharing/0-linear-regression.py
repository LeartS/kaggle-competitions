import math
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

if __name__ == '__main__':
    common_input_options = {'delimiter': ',', 'skiprows': 1,
                            'converters': {0: hour_from_dt_string} }
    train_dataset = load_data('data/train.csv', usecols=(0,1,2,3,4,5,6,7,8,11),
                              **common_input_options)
    estimator = linear_model.LinearRegression()
    k_fold = cross_validation.KFold(n=len(train_dataset), n_folds=10,
                                    indices=True)
    a = 0.0
    for train_indices, test_indices in k_fold:
        train_train_X = train_dataset[train_indices][:,:-1]
        train_train_y = train_dataset[train_indices][:,-1]
        train_test_X = train_dataset[test_indices][:,:-1]
        train_test_y = train_dataset[test_indices][:,-1]
        r = estimator.fit(train_train_X, train_train_y).predict(train_test_X)
        r = np.where(r > 0, r, 0.01)
        s = math.sqrt(metrics.mean_squared_error(np.log(train_test_y + 1), np.log(r + 1.0)))
        a += s
        print 'Score: {:.4f}'.format(s)
    print 'Average score: {:.4f}'.format(a/len(k_fold))
    test_dataset = load_data('data/test.csv', usecols=(0,1,2,3,4,5,6,7,8),
                             **common_input_options)
    common_input_options['converters'] = {}
    out_column = load_data('data/test.csv', usecols=(0,), dtype=str,
                           **common_input_options)
    results = estimator.fit(
        train_dataset[:,:-1], train_dataset[:,-1]
    ).predict(test_dataset)
    results = np.where(results > 0, results, 0.01).astype(np.int)
    save_data('data/out.csv', np.column_stack((out_column.T, results.T)),
              delimiter=',', header='datetime,count', fmt=('%s', '%s'),
              comments='')
