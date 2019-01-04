from collections import defaultdict

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer

from urlclustering.noise_feature import tokenize_url, digit_ratio, readability, special_char_ratio

cluster_columns = ['url', 'feature']


def _lazy_tokenize_url(urls, max_digital_ratio):
    for url in urls:
        yield tokenize_url(url, max_digital_ratio)


def _word_score(word, frequency):
    digit_score = frequency * (1 - digit_ratio(word))
    readability_score = frequency * readability(word)
    special_char_score = frequency * (1 - special_char_ratio(word))
    return int(digit_score + readability_score + special_char_score)


class FeatureCounter:
    def __init__(self, max_digital_ratio = 1, max_group_sampes = None):
        self.words = None
        self.max_digital_ratio = max_digital_ratio
        self.max_group_samples = max_group_sampes
        self.len_weight = 1

    def _fit(self, urls):
        self.len_weight = len(urls)
        corpus = [" ".join(url) for url in _lazy_tokenize_url(urls, self.max_digital_ratio)]
        counter = CountVectorizer(lowercase=True)
        x = counter.fit_transform(corpus)
        self.words = counter.vocabulary_
        return x


    def fit(self, urls):
        self.len_weight = len(urls)
        print('number of raw data: ', self.len_weight)
        # sample_urls = urls_preprocess(urls)
        sample_urls = urls
        total = len(sample_urls)
        print('number of sample data: ', total)
        _vocabulary = defaultdict()
        _vocabulary.default_factory = _vocabulary.__len__
        _word_counter = defaultdict(int)
        indices = []
        indptr = []
        values = []
        indptr.append(0)
        i = 0
        print(f'extracting feature, 0% complete', end='\r', flush=True)
        url_len_array = []
        for url in _lazy_tokenize_url(sample_urls, self.max_digital_ratio):
            url_len_array.append(len(url))
            local_word_counter = dict()
            for word in url:
                word_idx = _vocabulary[word]
                local_word_counter[word_idx] = 1
                _word_counter[word_idx] += 1

            indices.extend(local_word_counter.keys())
            values.extend(local_word_counter.values())
            indptr.append(len(indices))
            i += 1
            if i % 1000 == 0:
                print(f'extracting feature, {i/total:.0%} complete', end='\r', flush=True)

        print()
        print("consolidating data.....")
        indices = np.asarray(indices)
        indptr = np.asarray(indptr)
        values = np.asarray(values)

        print("transforming data.....")
        url_len_array = sp.csr_matrix(np.reshape(np.array(url_len_array), (-1, 1)))
        x = sp.csr_matrix((values, indices, indptr), shape=(len(indptr) - 1, len(_vocabulary)))

        x = sp.hstack([x, url_len_array])
        items = sorted(_vocabulary.items(), key=lambda y: y[1])
        self.words = [item[0] for item in items]
        score_array = [_word_score(word, _word_counter[idx]) for word, idx in items]
        score_array.append(self.len_weight)

        return sample_urls, sp.csr_matrix(x).multiply(score_array)

