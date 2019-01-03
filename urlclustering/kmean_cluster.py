import concurrent.futures
import sys
import warnings

import numpy as np
from scipy import sparse
from sklearn import cluster, metrics
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import CountVectorizer

from urlclustering.feature import tokenize
from urlclustering.report import log_report


class KmeanClustering:
    def __init__(self, max_n_clusters=40, max_iter=20):
        self.max_n_clusters = max_n_clusters
        self.max_iter = max_iter
        self.kmean = None
        self.count_vectorizer = None
        self.features = None
        self.clusters = None

    def _kmean(self, observations, n_clusters):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            result = cluster.KMeans(n_clusters=n_clusters, max_iter=self.max_iter).fit(observations)
        score = metrics.calinski_harabaz_score(observations.toarray(), result.labels_)
        return score, result

    def fit(self, urls):
        self.features = self._extract_features(urls)

        cluster_range = range(2, self.max_n_clusters)
        max_ch_score = 0
        best_k = 0
        with concurrent.futures.ProcessPoolExecutor() as executor:
            k_to_kmean_future = {k: executor.submit(self._kmean, self.features, k) for k in cluster_range}

            k_to_kmean = {}
            for k in cluster_range:
                ch_score, fit_result = k_to_kmean_future[k].result()
                k_to_kmean[k] = fit_result
                if ch_score > max_ch_score:
                    max_ch_score = ch_score
                    best_k = k

                print(f'k: {k}, Calinski Harabaz score: {str(ch_score)}, iterations: {fit_result.n_iter_}', flush=True)

        print("\nbest_k: ", best_k)
        self.kmean = k_to_kmean[best_k]
        self.clusters = [[] for _ in range(best_k)]
        for idx, label in enumerate(k_to_kmean[best_k].labels_):
            self.clusters[label].append(lines[idx])

        log_report(best_k, self.count_vectorizer.get_feature_names(), self.clusters)

    def predict(self, url):
        len_, tokens = tokenize(url)
        f = self.count_vectorizer.transform([" ".join(tokens)])
        feature = np.append(f.toarray(), len_)
        return self.kmean.predict([feature])

    def _extract_features(self, urls):
        len_array = []
        corpuses = []
        for url in urls:
            len_, tokens = tokenize(url)
            len_array.append(len_)
            corpuses.append(" ".join(tokens))

        self.count_vectorizer = CountVectorizer()
        freq_array = self.count_vectorizer.fiT_transform(corpuses)
        features = sparse.csr_matrix(
            np.concatenate((freq_array.toarray(), np.array(len_array).reshape(-1, 1)), axis=1)
        )
        return features


if __name__ == "__main__":
    with open(sys.argv[1]) as fp:
        lines = [line.rstrip() for line in fp.readlines()]

    np.set_printoptions(threshold=np.nan)
    kmean = KmeanClustering(max_n_clusters=int(sys.argv[2]))
    kmean.fit(lines)

    url = input("enter the url: ")
    while len(url) > 0:
        url = input("enter the url: ")
        group = kmean.predict(url)
        print(kmean.clusters[group[0]][0])
