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

def load_data(path, **kwargs):
    with open(path, 'r') as train_file:
        reader = csv.reader(train_file, delimiter='\t')
        dataset = [l for l in reader][1:]
    return dataset

def train(train_dataset):
    for pid, sid, text, score in train_dataset:
        text = text.replace(',', '').replace('.', '')
        phrases.append((int(pid), int(sid), text, int(score)))

    for a, b, phrase, score in phrases:
        score -= 2.0
        words = phrase.split()
        prev_modifier = False
        for word in words:
            if word in modifiers or modifiers_reg.match(word):
                prev_modifier = True
                mod_data[word] = 0.0
                continue
            if prev_modifier:
                continue
            if not word in word_data:
                word_data[word] = {'score': 0.0, 'count': 0.0}
            prev_modifier = False
            word_data[word]['score'] += score/len(words)
            word_data[word]['count'] += 1.0/len(words)

    for word in word_data.keys():
        word_data[word]['score'] /= float(word_data[word]['count'])

    for a, b, phrase, score in phrases:
        score -= 2.0
        words = phrase.split()
        modif = False
        for word in words:
            if word in modifiers or modifiers_reg.match(word):
                modif = word
                continue
            if modif and word in word_data:
                if modif == 'slightly':
                    print "{} {}: {} {} [{}]".format(modif, word, word_data[word]['score'], score, phrase)
                if word_data[word]['score'] * score > 0.0: # same sign
                    mod_data[modif] += math.fabs(word_data[word]['score'] - score)
                else:
                    mod_data[modif] -= math.fabs(word_data[word]['score'] - score)
                modif = False

def fit(test):
    results = []
    for pid, sid, phrase in test:
        total = 0.0
        total_weight = 0.0
        for word in phrase.split():
            if len(word) > 2:
                if word in words:
                    total += words[word]['score'] * math.fabs(words[word]['score'])
                    total_weight += math.fabs(words[word]['score'])
        if total_weight:
            results.append(total/float(total_weight))
        else:
            results.append(0)
    return results

def output(results, header_row):
    with open('output.csv', 'w') as output_file:
        writer = csv.writer(output_file)
        writer.writerow(header_row)
        writer.writerows(results)

if __name__ == '__main__':
    train_dataset = load_data('data/train.tsv')
    k_fold = cross_validation.KFold(n=len(train_dataset), n_folds=10, indices=True)
    for train_indices, test_indices in k_fold:
        print 'Train: {}, Test: {}'.format(train_indices, test_indices)
        print type(train_indices)
    raise SystemExit(0)
    train(train_dataset)
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
