import argparse
import concurrent.futures
import sys
import warnings
from collections import defaultdict

import numpy as np
from sklearn import cluster, metrics
from sklearn.exceptions import ConvergenceWarning

from urlclustering.cluster_feature import FeatureCounter
from urlclustering.noise_feature import tokenize_url
from urlclustering.report import log_report


class KmeanClustering:
    def __init__(self, max_n_clusters=40, max_iter=40):
        self.max_n_clusters = max_n_clusters
        self.max_iter = max_iter
        self.kmean = None
        self.count_vectorizer = None
        self.clusters = None
        self.best_k = 4
        self.words = None

    def _kmean(self, observations, n_clusters):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            result = cluster.MiniBatchKMeans(n_clusters=n_clusters, max_iter=self.max_iter,
                                             batch_size=1000).fit(observations)
        score = metrics.calinski_harabaz_score(observations.toarray(), result.labels_)
        return score, result

    def fit(self, urls):
        print("collection sample and featuring...")
        feature_counter = FeatureCounter()
        X = feature_counter.fit(urls)
        self.words = feature_counter.words

        cluster_range = range(2, self.max_n_clusters)
        max_ch_score = 0
        with concurrent.futures.ProcessPoolExecutor() as executor:
            k_to_kmean_future = {k: executor.submit(self._kmean, X, k) for k in cluster_range}

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
        self.clusters = defaultdict(list)
        for idx, label in enumerate(k_to_kmean[self.best_k].labels_):
            self.clusters[label].append(urls[idx])

    def predict(self, url):
        len_, tokens = tokenize_url(url)
        f = self.count_vectorizer.transform([" ".join(tokens)])
        feature = np.append(f.toarray(), len_)
        return self.kmean.predict([feature])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_clusters', type=int, default=20, help='max clusters')
    parser.add_argument('inputfile', nargs='?', type=argparse.FileType('r'), default=sys.stdin,
                        help='input file name')
    args = parser.parse_args()

    lines = [line.strip() for line in args.inputfile.readlines()]
    np.set_printoptions(threshold=np.nan)

    print("running kmean...")
    kmean = KmeanClustering(max_n_clusters=args.max_clusters)
    kmean.fit(lines)

    log_report(kmean.best_k, kmean.words, kmean.clusters)
    # url = input("enter the url: ")
    # while len(url) > 0:
    #     url = input("enter the url: ")
    #     group = kmean.predict(url)
    #     print(kmean.clusters[group[0]][0])
