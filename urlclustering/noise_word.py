import pickle

import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.utils.fixes import signature

from urlclustering.dictionary import Dictionary
from urlclustering.storage import create_session, Sample
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
        self._clf = svm.LinearSVC()
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
        return np.array(
            [self._pos(url, word), len(word), self._readability(word), digit_ratio(word)]
        )

    def _pos(self, url, word):
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
        while i < word_len - 3:
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
    if len(word) == 0:
        return 0

    digit_count = 0
    for c in word:
        if '0' <= c <= '9':
            digit_count += 1
    return digit_count / len(word)


if __name__ == '__main__':
    x = []
    y = []
    print('gathering training data')
    with create_session() as session:
        for sample in session.query(Sample).filter(Sample.label != None):
            x.append(pickle.loads(sample.features))
            y.append(int(sample.label))

    print('splitting training and test data')
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    print('training')
    detector = NoiseWordDetector()
    detector.fit(x_train, y_train)

    print('analyze result')
    y_score = detector.predict(x_test)
    average_precision = average_precision_score(y_test, y_score)
    print(f'Average precision-recall score: {average_precision:0.2f}')
    precision, recall, _ = precision_recall_curve(y_test, y_score)

    step_kwargs = ({'step': 'post'} if 'step' in signature(plt.fill_between).parameters else {})
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))



    # while(True):
    #     url = input('url: ')
    #     word = input('word: ')
    #     print(detector.predict(url, word))
