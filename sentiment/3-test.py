import math
import numpy
from sklearn import cross_validation
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score

class CustomEstimator(BaseEstimator):

    def __init__(self):
        self.words = {}

    def reset(self):
        self.words.clear()

    def fit(self, X, y):
        if X.shape[0] != y.shape[0]:
            raise ValueError, 'X and y should have the same length'
        self.reset()
        for text, score in zip(X,y):
            text = text.replace(',', '').replace('.', '')
            score = int(score) - 2.0
            words = text.split()
            for word in words:
                if not word in self.words:
                    self.words[word] = {'score': 0.0, 'count': 0.0}
                self.words[word]['score'] += score / len(words)**2
                self.words[word]['count'] += 1.0 / len(words)**2
        for word in self.words.keys():
            self.words[word]['score'] /= float(self.words[word]['count'])
        return self

    def predict(self, T):
        results = []
        for phrase in T:
            total = 0.0
            total_weight = 0.0
            for word in phrase.split():
                if len(word) > 2:
                    if word in self.words:
                        word_score = self.words[word]['score']
                        total += word_score * math.fabs(word_score)
                        total_weight += 1.0 * math.fabs(word_score)
            if total_weight:
                results.append(int(round(total/float(total_weight))) + 2)
            else:
                results.append(2.0)
        return numpy.array(results, dtype=numpy.int8)

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

def load_data(path, **kwargs):
    return numpy.loadtxt(path, **kwargs)

def save_data(path, data, **kwargs):
    numpy.savetxt(path, data, **kwargs)

if __name__ == '__main__':
    # the strange comments parameter is a hacky workaround for
    # https://github.com/numpy/numpy/issues/5155
    train_dataset = load_data('data/train.tsv', delimiter='\t', skiprows=1,
                              usecols=(0,2,3), comments='kl4kk4k', dtype=object)
    estimator = CustomEstimator()
    k_fold = cross_validation.KFold(n=len(train_dataset), n_folds=5,
                                    indices=True)
    a = 0.0
    for train_indices, test_indices in k_fold:
        train_train_X = train_dataset[train_indices][:,1]
        train_train_y = train_dataset[train_indices][:,-1].astype(numpy.int8)
        train_test_X = train_dataset[test_indices][:,1]
        train_test_y = train_dataset[test_indices][:,-1].astype(numpy.int8)
        
        s = estimator.fit(train_train_X, train_train_y).score(train_test_X,
                                                              train_test_y)
        a += s
        print 'Score: {:.4f}'.format(s)
    print 'Average score: {:.4f}'.format(a/len(k_fold))
    test_dataset = load_data('data/test.tsv', delimiter='\t', skiprows=1,
                             usecols=(0,2), comments=None, dtype=object)
    results = estimator.fit(
        train_dataset[:,1], train_dataset[:,-1].astype(numpy.int8)
    ).predict(test_dataset[:,-1])
    save_data('data/out.csv', numpy.column_stack((test_dataset[:,0], results)),
              delimiter=',', header='PhraseId,Sentiment', fmt=('%s', '%u'),
              comments='')
