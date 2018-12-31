import argparse
import pickle
import sys
import time

from urlclustering.noise_word import NoiseWordDetector
from urlclustering.storage import Sample, create_session
from urlclustering.utils import tokenize_url


def process_raw_samples(inputfile, truncate=False):
    samples_dict = {}
    print('collecting data....')
    urls = inputfile.readlines()
    total = len(urls)
    i = 0
    for url in urls:
        url = url.strip()
        for word in tokenize_url(url):
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
    for word in samples_dict.keys():
        value = samples_dict[word]
        sample = Sample(word=word, url=value[0], frequency=value[1]/total, amount=value[1])
        sample.features = detector.extract_features(sample)
        samples.append(sample)
        i += 1
        if i % 10000 == 0:
            print(f'extracting features, {i/data_len:.0%} complete', end='\r')
    print('extracting features, 100% complete')

    i = 0
    data_len = len(samples)
    with create_session(expire_on_commit=False) as session:
        if truncate:
            Sample.truncate_table(Sample, session)
            session.commit()
        for sample in samples:
            sample.features = pickle.dumps(sample.features)
            session.add(sample)
            i += 1
            if i % 10000 == 0:
                session.commit()
                print(f'writing samples to storage, {i/data_len:.0%} complete', end='\r')
    print('writing samples to storage, 100% complete')


def update_all_features():
    detector = NoiseWordDetector()
    with create_session() as session:
        samples = session.query(Sample).all()

        data_len = len(samples)
        i = 0
        for sample in samples:
            sample.features = pickle.dumps(detector.extract_features(sample))
            i += 1
            if i % 10000 == 0:
                print(f'update feature, {i/data_len:.0%} complete', end='\r')
                session.commit()
        session.commit()
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--truncate', action='store_true', default=False,
                        help='truncate all data before processing')
    parser.add_argument('--feature', action='store_true', default=False,
                        help='update feature')
    parser.add_argument('inputfile', nargs='?', type=argparse.FileType('r'),
                        default=sys.stdin, help='input file name')

    args = parser.parse_args()

    action = None
    if args.feature:
        def action(): update_all_features()
    else:
        def action(): process_raw_samples(args.inputfile, args.truncate)

    start = time.time()
    try:
        action()
    finally:
        args.inputfile.close()

    end = time.time()
    print(f'elapsed {end-start} secs')
