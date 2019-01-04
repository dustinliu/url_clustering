import concurrent.futures
import sys
import warnings

import numpy as np
from scipy import sparse
from sklearn import cluster, metrics
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import CountVectorizer

from urlclustering.cluster_feature import UrlclusterFeature
from urlclustering.noise_feature import tokenize_url
from urlclustering.report import log_report


class KmeanClustering:
    def __init__(self, max_n_clusters=40, max_iter=10):
        self.max_n_clusters = max_n_clusters
        self.max_iter = max_iter
        self.kmean = None
        self.count_vectorizer = None
        self.clusters = None
        self.best_k = 4

    def _kmean(self, observations, n_clusters):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            result = cluster.KMeans(n_clusters=n_clusters, max_iter=self.max_iter, )\
                .fit(observations)
        score = metrics.calinski_harabaz_score(observations.toarray(), result.labels_)
        return score, result

    def fit(self, X):
        cluster_range = range(2, self.max_n_clusters)
        max_ch_score = 0
        with concurrent.futures.ProcessPoolExecutor() as executor:
            k_to_kmean_future = {k: executor.submit(self._kmean, X, k)
                                 for k in cluster_range}

            k_to_kmean = {}
            for k in cluster_range:
                ch_score, fit_result = k_to_kmean_future[k].result()
                k_to_kmean[k] = fit_result
                if ch_score > max_ch_score:
                    max_ch_score = ch_score
                    self.best_k = k

                print(f'k: {k:<3} Calinski Harabaz score: {ch_score:<13.2f}', end='')
                print(f' iterations: {fit_result.n_iter_}', flush=True)

        print("\nbest_k: ", self.best_k)
        self.kmean = k_to_kmean[self.best_k]
        self.clusters = [[] for _ in range(self.best_k)]
        for idx, label in enumerate(k_to_kmean[self.best_k].labels_):
            self.clusters[label].append(lines[idx])


    def predict(self, url):
        len_, tokens = tokenize_url(url)
        f = self.count_vectorizer.transform([" ".join(tokens)])
        feature = np.append(f.toarray(), len_)
        return self.kmean.predict([feature])


if __name__ == "__main__":
    with open(sys.argv[1]) as fp:
        lines = [line.strip() for line in fp.readlines()]

    print("collection sample and featuring...")
    np.set_printoptions(threshold=np.nan)
    feature_counter = UrlclusterFeature()
    X = feature_counter.fit(lines)

    print("running kmean...")
    kmean = KmeanClustering(max_n_clusters=int(sys.argv[2]))
    kmean.fit(X)

    log_report(kmean.best_k, feature_counter._vocabulary.keys(), kmean.clusters)
    # url = input("enter the url: ")
    # while len(url) > 0:
    #     url = input("enter the url: ")
    #     group = kmean.predict(url)
    #     print(kmean.clusters[group[0]][0])
