import math
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer

import urlclustering.noise_feature as nf
from urlclustering.util import ending_print, elapsed_time, execute_fork_join

cluster_columns = ['url', 'feature']


def _lazy_tokenize_url(urls, max_digital_ratio):
    for url in urls:
        yield nf.tokenize_url(url, max_digital_ratio)


def _word_score(word, frequency):
    return math.ceil((1 + nf.readability(word) - nf.special_char_ratio(word) -
                        nf.digit_ratio(word)) * frequency)


class FeatureCounter:
    def __init__(self, max_digital_ratio=1):
        self.words = None
        self.max_digital_ratio = max_digital_ratio
        self._word_count = defaultdict(int)

    def _fit(self, urls):
        self.len_weight = len(urls)
        corpus = [" ".join(url) for url in _lazy_tokenize_url(urls, self.max_digital_ratio)]
        counter = CountVectorizer(lowercase=True)
        x = counter.fit_transform(corpus)
        self.words = counter.vocabulary_
        return x

    def fit(self, dataSet):
        total = dataSet.shape[0]
        print('number of samples: ', total)
        _vocabulary = defaultdict()
        _vocabulary.default_factory = _vocabulary.__len__
        indices = []
        indptr = []
        values = []
        indptr.append(0)
        i = 0
        print(f'extracting feature, 0% complete', end='\r', flush=True)
        start_time = time.time()
        url_len_array = []
        for url in _lazy_tokenize_url(dataSet['url'].tolist(), self.max_digital_ratio):
            url_len_array.append(len(url) * total)
            local_word_count = dict()
            for word in url:
                word_idx = _vocabulary[word]
                local_word_count[word_idx] = 1
                self._word_count[word_idx] += 1

            indices.extend(local_word_count.keys())
            values.extend(local_word_count.values())
            indptr.append(len(indices))
            i += 1
            if i % 1000 == 0:
                print(f'extracting feature, {i/total:.0%} complete', end='\r', flush=True)

        ending_print(f'feature extracting done, {elapsed_time(time.time() - start_time)} elapsed')

        indices = np.asarray(indices)
        indptr = np.asarray(indptr)
        values = np.asarray(values)

        url_len_array = \
            sp.csr_matrix(np.reshape(np.array(url_len_array), (-1, 1)))
        x = sp.csr_matrix((values, indices, indptr), shape=(indptr.shape[0] - 1, len(_vocabulary)))

        print("weight calculating......", end='\r')
        start_time = time.time()
        items = sorted(_vocabulary.items(), key=lambda y: y[1])
        self.words = [item[0] for item in items]
        # score_array = [_word_score(word, self._word_count[idx]) for word, idx in items]
        score_array = execute_fork_join(self._score, items, batch_size=100000)
        x = x.multiply(score_array)
        x = sp.hstack([x, url_len_array], format='csr')
        ending_print(f'weighting done, {elapsed_time(time.time()-start_time)} elapsed')

        start_time = time.time()
        j = list(x)
        print(f'make dataSet time: {elapsed_time(time.time() - start_time)}')
        dataSet['feature'] = pd.Series(list(j))
        return dataSet

    def _score(self, items):
        return [_word_score(item[0], self._word_count[item[1]]) for item in items]
