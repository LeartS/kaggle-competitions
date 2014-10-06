import math
import numpy as np
from sklearn import cross_validation
from sklearn import linear_model
from sklearn import metrics

def load_data(path, **kwargs):
    return np.loadtxt(path, **kwargs)

def save_data(path, data, **kwargs):
    np.savetxt(path, data, **kwargs)

def log_arr(array):
    return 

if __name__ == '__main__':
    train_dataset = load_data('data/train.csv', delimiter=',', skiprows=1,
                              usecols=(1,2,3,4,5,6,7,8,11))
    estimator = linear_model.LinearRegression()
    k_fold = cross_validation.KFold(n=len(train_dataset), n_folds=5,
                                    indices=True)
    a = 0.0
    for train_indices, test_indices in k_fold:
        train_train_X = train_dataset[train_indices][:,:-1]
        train_train_y = train_dataset[train_indices][:,-1]
        train_test_X = train_dataset[test_indices][:,:-1]
        train_test_y = train_dataset[test_indices][:,-1]
        r = estimator.fit(train_train_X, train_train_y).predict(train_test_X)
        r = np.where(r > 0, r, 0.01)
        s = metrics.mean_squared_error(np.log(train_test_y + 1), np.log(r + 1.0))
        a += s
        # mr = np.mean(train_train_y)
        # ms = metrics.mean_squared_error(np.log(train_test_y + 1), np.log(mr + 1.0))
        print 'Score: {:.4f}'.format(s)
        # print 'Mean value score: {:.4f}'.format(ms)
    print 'Average score: {:.4f}'.format(a/len(k_fold))
    # 1/0
    test_dataset = load_data('data/test.tsv', delimiter='\t', skiprows=1,
                             usecols=(0,2), comments=None, dtype=object)
    results = estimator.fit(
        train_dataset[:,1], train_dataset[:,-1].astype(np.int8)
    ).predict(test_dataset[:,-1])
    save_data('data/out.csv', np.column_stack((test_dataset[:,0], results)),
              delimiter=',', header='PhraseId,Sentiment', fmt=('%s', '%u'),
              comments='')
