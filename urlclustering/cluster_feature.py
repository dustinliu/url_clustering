from collections import defaultdict

import numpy as np
import scipy.sparse as sp

from urlclustering.noise_feature import tokenize_url, digit_ratio, readability, special_char_ratio

cluster_columns = ['url', 'feature']


def _lazy_tokenize_url(urls):
    for url in urls:
        yield tokenize_url(url)


def _word_score(word, frequency):
    digit_score = frequency * (1 - digit_ratio(word))
    readability_score = frequency * readability(word)
    special_char_score = frequency * (1 - special_char_ratio(word))
    return digit_score + readability_score + special_char_score


class UrlclusterFeature:
    def __init__(self):
        self._vocabulary = defaultdict()
        self._vocabulary.default_factory = self._vocabulary.__len__
        self._word_counter = defaultdict(int)

    def fit(self, urls):
        indices = []
        indptr = []
        values = []
        indptr.append(0)
        for url in _lazy_tokenize_url(urls):
            local_word_counter = dict()
            for word in url:
                word_idx = self._vocabulary[word]
                local_word_counter[word_idx] = 1
                self._word_counter[word_idx] += 1

            indices.extend(local_word_counter.keys())
            values.extend(local_word_counter.values())
            indptr.append(len(indices))

        indices = np.asarray(indices)
        indptr = np.asarray(indptr)
        values = np.asarray(values)

        x = sp.csr_matrix((values, indices, indptr), shape=(len(indptr) - 1, len(self._vocabulary)))

        items = sorted(self._vocabulary.items(), key=lambda y: y[1])
        score_array = [_word_score(word, self._word_counter[idx]) for word, idx in items]

        return x.multiply(score_array)
