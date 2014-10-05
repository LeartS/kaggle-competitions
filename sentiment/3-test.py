# = IDEA =
# Simple solution: give a 'score' to each word based on the average
# sentiment of the phrases it's in.
# Calculate the sentiment on new phrases by calculating the average score
# of its words, weighted by the "non-neutralness" of the word.
#
# = SCORE ON KAGGLE PUBLIC LEADERBOARD =
# 0.53859

import csv
import math
import re
from sklearn import cross_validation

modifiers = """
no not somewhat too little less more quite very so never ever seldom
always often still
"""
modifiers = modifiers.split()
modifiers_reg = re.compile('^[a-zA-Z]{3,}ly$')

phrases = []
word_data = {}
mod_data = {}

class Estimator(object):

    def __init__(self):
        self.word_data = {}
        self.phrases = []

    def reset(self):
        self.word_data.clear()
        self.phrases = []

    def fit(X, Y):
        self.reset()
        for pid, sid, text, score in zip(X,Y):
            text = text.replace(',', '').replace('.', '')
            phrases.append((int(pid), int(sid), text, int(score)))

        for a, b, phrase, score in phrases:
            score -= 2.0
            words = phrase.split()
            for word in words:
                if not word in word_data:
                    word_data[word] = {'score': 0.0, 'count': 0.0}
                word_data[word]['score'] += score/len(words)**3
                word_data[word]['count'] += 1.0/len(words)**3

        for word in word_data.keys():
            word_data[word]['score'] /= float(word_data[word]['count'])

        return self

    def predict(T):
        results = []
        for pid, sid, phrase in test:
            total = 0.0
            total_weight = 0.0
            for word in phrase.split():
                if len(word) > 2:
                    if word in word_data:
                        total += word_data[word]['score'] * math.fabs(word_data[word]['score'])
                        total_weight += math.fabs(word_data[word]['score'])
            if total_weight:
                results.append(str(int(round(total/float(total_weight))) + 2))
            else:
                results.append(0 + 2)
        return results 

def load_data(path, **kwargs):
    with open(path, 'r') as train_file:
        reader = csv.reader(train_file, delimiter='\t')
        dataset = [l for l in reader][1:]
    return dataset

def fit(train_dataset):

def score(results, reference):
    return len([r for r, f in zip(results, reference) if r == f])/float(len(results))

def output(results, header_row):
    with open('output.csv', 'w') as output_file:
        writer = csv.writer(output_file)
        writer.writerow(header_row)
        writer.writerows(results)

if __name__ == '__main__':
    train_dataset = load_data('data/train.tsv')
    k_fold = cross_validation.KFold(n=len(train_dataset), n_folds=5, indices=True)
    a = 0.0
    for train_indices, test_indices in k_fold:
        train_train = [train_dataset[i] for i in train_indices]
        train_test = [train_dataset[i][:-1] for i in test_indices]
        train_ref = [train_dataset[i][-1] for i in test_indices]
        
        fit(train_train)
        results = predict(train_test)
        s = score(results, train_ref)
        a += s
        print 'Score: {}'.format(score(results, train_ref))
        # print 'Train: {}, Test: {}'.format(train_indices, test_indices)
    print 'Average score: {}'.format(a/len(k_fold))
    raise SystemExit(0)
    raise SystemExit(0)
    test_dataset = load_data('data/test.tsv')
    results = fit(test_dataset)
    output(
        list(zip(
            map(lambda x: x[0], test_dataset),
            map(lambda x: int(round(x))+2, results)
        )),
        ('PhraseId', 'Sentiment')
    )
