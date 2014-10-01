# Simple solution:
# train: Calculate the average sentiment for every word, considering only sentences
# fit: calculate the average sentiment of the words in that phrase.
# assument sentiment = 2 when you encounter words not in the train data

import numpy
import sklearn
import csv
import math

dataset = None
output = None
sentences = []
phrases = []
words = {}

def load_data(path, **kwargs):
    with open(path, 'r') as train_file:
        reader = csv.reader(train_file, delimiter='\t')
        dataset = [l for l in reader][1:]
    return dataset

def train(train_dataset):
    prev_sid = 0
    for pid, sid, text, score in train_dataset:
        phrases.append((int(pid), int(sid), text, int(score)))
        if int(sid) != prev_sid:
            prev_sid = int(sid)
            sentences.append((int(pid), int(sid), text, int(score)))

    for a, b, phrase, score in phrases:
        score -= 2
        for word in phrase.split():
            if word in words:
                words[word]['score'] += score
                words[word]['count'] += 1
            else:
                words[word] = {'score': score, 'count': 1}

    for word in words.keys():
        words[word]['score'] /= float(words[word]['count'])

def fit(test):
    results = []
    for pid, sid, phrase in test:
        total = 0.0
        total_weight = 0.0
        for word in phrase.split():
            if len(word) > 2:
                if word in words:
                    total += words[word]['score'] * math.fabs(words[word]['score'])#words[word]['count']
                    total_weight += math.fabs(words[word]['score'])#words[word]['count']

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
    input_options = {'comments': None, 'skip_header': 1, 'delimiter': '\t',
                     'dtype': numpy.uint32}
    train_dataset = load_data('data/train.tsv', **input_options)
    train(train_dataset)
    test_dataset = load_data('data/test.tsv', **input_options)
    results = fit(test_dataset)
    output(
        list(zip(map(lambda x: x[0], test_dataset), map(lambda x: int(round(x))+2, results))),
        ('PhraseId', 'Sentiment')
    )
