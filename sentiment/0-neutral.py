# Neutral solution: assign 2 to all phrases

import numpy
import sklearn
import csv

dataset = None
output = None

def load_data(path, **kwargs):
    with open(path, 'r') as train_file:
        dataset = numpy.genfromtxt(train_file, **kwargs)
    return dataset

def train(train_dataset):
    pass

def fit(test):
    # Append a column of 2
    # Suprisingly, np.empty + fill is actually faster than full.
    # We use full anyway because clearer.
    return numpy.concatenate((
        test,
        numpy.full((test.shape[0], 1), 2, dtype=numpy.uint32)
    ), 1)

def output(results, header_row):
    with open('data/output.csv', 'w') as output_file:
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
    output(results[:,(0,-1)], ('PhraseId', 'Sentiment'))
