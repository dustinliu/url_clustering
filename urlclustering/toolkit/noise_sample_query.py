import argparse
import pickle

from urlclustering.storage import create_session, Sample

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--word', help='query by word')
    parser.add_argument('--id', help='query by id')
    parser.add_argument('--column', nargs='+', help='column to show')

    args = parser.parse_args()

    if args.word and args.id:
        parser.error("query word and id in the same time is not allowed")

    if args.word:
        with create_session() as session:
            for sample in session.query(Sample).filter_by(word=args.word):
                print(sample.id, sample.url, pickle.loads(sample.features))

    if args.id:
        with create_session() as session:
            for sample in session.query(Sample).filter_by(word=args.id).all():
                print(sample.id, sample.url, pickle.loads(sample.features))

