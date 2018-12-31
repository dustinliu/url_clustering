import argparse

from urlclustering.noise_word import digit_ratio
from urlclustering.storage import Sample, create_session

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', dest='ratio', help='filter digital word')
    parser.add_argument('-l', dest='length', help='filter long word')

    args = parser.parse_args()

    if args.ratio:
        filter = lambda x: digit_ratio(x) < float(args.ratio)
    elif args.length:
        filter = lambda x: len(x) > int(args.length)
    else:
        filter = lambda _: True

    with create_session() as session:
        print('=========================================')
        for sample in session.query(Sample).filter(Sample.label == True):
            if filter(sample.word):
                print(sample.id, sample.word, sample.amount, sample.url)
