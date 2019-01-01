import argparse
import sys
import time

import pandas as pd

from urlclustering.noise_word import NoiseWordDetector
from urlclustering.utils import tokenize_url

pd_columns = ['word', 'url', 'frequency', 'amount', 'position', 'length', 'readability', 'digital_ratio', 'label']


def write_sample(samples, file):
    samples.to_csv(file, sep='\t')


def read_sample(file):
    return pd.read_csv(file, sep='\t', index_col=0)


def process_raw_samples(inputfile, outputfile):
    samples_dict = {}
    print('collecting data....')
    urls = inputfile.readlines()
    total = len(urls)
    i = 0
    for url in urls:
        url = url.strip()
        for word in tokenize_url(url):
            if word is None or len(word) == 0:
                continue

            word = word.lower()
            if word in samples_dict:
                samples_dict[word][1] += 1
            else:
                samples_dict[word] = [url, 1]
        i += 1
        if i % 10000 == 0:
            print(f'processing samples, {i/total:.0%} complete', end='\r')
    print('processing samples, 100% complete')

    samples = []
    detector = NoiseWordDetector()
    i = 0
    data_len = len(samples_dict)
    print('extracting features, 0% complete', end='\r')
    for word in samples_dict.keys():
        value = samples_dict[word]
        feature = detector.extract_features(word, value[0])
        samples.append([word, value[0], value[1]/total, value[1],
                        feature[0], feature[1], feature[2], feature[3], None])
        i += 1
        if i % 10000 == 0:
            print(f'extracting features, {i/data_len:.0%} complete', end='\r')
    print('extracting features, 100% complete')

    print("write samples to storage")
    write_sample(pd.DataFrame(samples, columns=pd_columns), outputfile)


def update_all_features(file):
    detector = NoiseWordDetector()
    samples = read_sample(file)
    samples['feature'] = samples.apply(lambda row: detector.extract_features(row['word'], row['url']), axis=1)
    write_sample(file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--truncate', action='store_true', default=False,
                        help='truncate all data before processing')
    parser.add_argument('--feature', action='store_true', default=False,
                        help='update feature')
    parser.add_argument('inputfile', nargs='?', type=argparse.FileType('r'),
                        default=sys.stdin, help='input file name')
    parser.add_argument('outputfile', nargs='?', type=argparse.FileType('w'),
                        default=sys.stdout, help='output file name')

    args = parser.parse_args()

    action = None
    if args.feature:
        def action(): update_all_features()
    else:
        def action(): process_raw_samples(args.inputfile, args.outputfile)

    start = time.time()
    try:
        action()
    finally:
        args.inputfile.close()

    end = time.time()
    print(f'elapsed {end-start} secs')
