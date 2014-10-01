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

phrases = []
words = {}

def load_data(path, **kwargs):
    with open(path, 'r') as train_file:
        reader = csv.reader(train_file, delimiter='\t')
        dataset = [l for l in reader][1:]
    return dataset

def train(train_dataset):
    for pid, sid, text, score in train_dataset:
        phrases.append((int(pid), int(sid), text, int(score)))

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
    train(train_dataset)
    test_dataset = load_data('data/test.tsv')
    results = fit(test_dataset)
    output(
        list(zip(
            map(lambda x: x[0], test_dataset),
            map(lambda x: int(round(x))+2, results)
        )),
        ('PhraseId', 'Sentiment')
    )
