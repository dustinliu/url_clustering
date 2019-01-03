import argparse
import sys
from collections import Counter

import numpy as np
import pickle
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import train_test_split

from urlclustering.toolkit.noise_sample import read_sample


class NoiseWordDetector:
    features_columes = ['position', 'length', 'readability', 'digital_ratio']

    def __init__(self, model_file=None):
        if model_file is None:
            self._clf = LogisticRegression(random_state=13, solver='liblinear')
        else:
            self._clf = self._load(model_file)

    def _load(self, model_file):
        if hasattr(model_file, 'read'):
            self._clf = pickle.load(model_file)
        else:
            with open(model_file, 'r') as fp:
                self._clf = pickle.load(fp)

    def fit(self, x, y):
        self._clf.fit(x, y)

    def predict(self, x):
        return self._clf.predict(x)

    def save(self, dest_file):
        if hasattr(dest_file, 'write'):
            pickle.dump(self._clf, dest_file)
        else:
            with open(dest_file, 'w') as fp:
                pickle.dump(self._clf, fp)


def transform_feature(samples):
    return np.array(samples[NoiseWordDetector.features_columes].values.tolist()),\
        np.array(samples['label'].values.tolist())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('trainingfile', nargs='?', type=argparse.FileType('r'), default=sys.stdin,
                        help='training file name')
    parser.add_argument('testfile', nargs='?', type=argparse.FileType('r'), help='test file name')
    args = parser.parse_args()

    print('collecting training data')
    samples = read_sample(args.trainingfile)
    X_train, y_train = transform_feature(samples)

    print(f'Original dataset shape {Counter(y_train)}')
    print('resampling by smote')
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f'resampled dataset shape {Counter(y_res)}')

    if args.testfile:
        print("collecting test data")
        samples = read_sample(args.testfile)
        X_test, y_test = transform_feature(samples)
    else:
        print("no test data given, split from training data")
        X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.25)

    print('training.....')
    detector = NoiseWordDetector()
    detector.fit(X_train, y_train)

    print('\nanalyze result')
    y_score = detector.predict(X_test)
    average_precision = average_precision_score(y_test, y_score)
    print(f'Average precision-recall score: {average_precision:0.2f}')
    precision, recall, threshold = precision_recall_curve(y_test, y_score)
    print(f'precision\t\t{precision}')
    print(f'recall\t\t{recall}')
    print(f'threshold\t\t{recall}')

    # step_kwargs = ({'step': 'post'} if 'step' in signature(plt.fill_between).parameters else {})
    # plt.step(recall, precision, color='b', alpha=0.2, where='post')
    # plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.ylim([0.0, 1.05])
    # plt.xlim([0.0, 1.0])
    # plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))

    # pylab.show()

    # while(True):
    #     url = input('url: ')
    #     word = input('word: ')
    #     print(f'{detector.predict([extract_features(word, url)])}')
