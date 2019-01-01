from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pylab
from imblearn.over_sampling import SMOTE
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.utils.fixes import signature

from urlclustering.dictionary import Dictionary
from urlclustering.utils import tokenize_url


class NoiseWordDetector:
    dict_ = Dictionary()

    def __init__(self):
        self._clf = None

    def is_noise_word(self, word):
        if word.lower() in self.dict_:
            return False

        return True

    def fit(self, x, y):
        self._clf = MLPClassifier(alpha=1)
        self._clf.fit(x, y)

    # def predict(self, url, word):
    #     feature = self.extract_features(word, url)
    #     print(feature)
    #     return self._clf.predict([feature])

    def predict(self, x):
        return self._clf.predict(x)

    def extract_features(self, word, url):
        # pos, length, readability, digital_ratio
        word = word.lower()
        return (self._pos(word, url), len(word), self._readability(word), digit_ratio(word))

    def _pos(self, word, url):
        tokens = tokenize_url(url)
        return tokens.index(word) / len(tokens)

    def _readability(self, word):
        if len(word) == 0:
            return 1

        len_ = 0
        for token in self._tokenize(word):
            len_ += len(token)
        return len_ / len(word)


    def _tokenize(self, word):
        tokens = []
        word_len = len(word)
        i = 0
        while i <= word_len - 3:
            token_len = word_len - i
            pos = 45 if token_len > 45 else token_len  # the length of the longest english word is 45
            for j in range(pos, 2, -1):
                token = word[i:i+j]
                if token in self.dict_:
                    tokens.append(token)
                    i += len(token) - 1
                    break
            i += 1
        return tokens

def digit_ratio(word):
    try:
        if word == None or len(word) == 0:
            return 0
    except TypeError:
        print(f'type error: {word}, type: {type(word)}')
        return 0

    digit_count = 0
    for c in word:
        if '0' <= c <= '9':
            digit_count += 1
    return digit_count / len(word)


if __name__ == '__main__':
    X = []
    y = []
    print('gathering training data')
    samples = pd.read_csv('data/samples.csv', sep='\t', index_col=0)
    for index, sample in samples.iterrows():
        X.append(sample.feature)
        y.append(int(sample.label))

    X = np.array(X)
    y = np.array(y)

    print(f'Original dataset shape {Counter(y)}')
    print('resampling by smote')
    sm = SMOTE(random_state=42)
    # sm = SMOTEENN(random_state=0)
    X_res, y_res = sm.fit_resample(X, y)
    print(f'Resampled dataset shape {Counter(y_res)}')

    print('splitting training and test data')
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.25)

    print('training')
    detector = NoiseWordDetector()
    detector.fit(X_train, y_train)

    print('analyze result')
    y_score = detector.predict(X_test)
    average_precision = average_precision_score(y_test, y_score)
    print(f'Average precision-recall score: {average_precision:0.2f}')
    precision, recall, _ = precision_recall_curve(y_test, y_score)
    print(f'precision {precision}')
    print(f'recall{recall}')

    step_kwargs = ({'step': 'post'} if 'step' in signature(plt.fill_between).parameters else {})
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))

    pylab.show()
