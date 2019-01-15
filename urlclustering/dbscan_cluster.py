import argparse
import concurrent.futures
from collections import defaultdict

import numpy as np
from sklearn import cluster

from urlclustering.cluster_feature import FeatureCounter
from urlclustering.report import log_report


class DbscanClustering:
    def __init__(self, batch_size=10000):
        self._batch_size = batch_size
        self.n_clusters = None
        self.labels = None
        self.clusters = defaultdict(list)
        self.features = None

    def _fit(self, x):
        print(f'dbscan clustering {x.shape[0]} samples')
        db_ = cluster.DBSCAN(eps=50, min_samples=2).fit(x)
        # return n_clusters and labels
        return len(set(self.labels)) - (1 if -1 in self.labels else 0), db_.labels_

    def fit(self, urls):
        print("collecting data...")
        feature_counter = FeatureCounter()
        x = feature_counter.fit(urls)
        self.features = feature_counter.words

        # batch_pool = np.array_split(x.toarray(), int(x.shape[0]/self._batch_size))
        batch_pool = self.slice_sample(x)
        with concurrent.futures.ProcessPoolExecutor() as executor:
            future_list = executor.map(self._fit, batch_pool)

        for idx, label in enumerate(self.labels):
            self.clusters[label].append(x[idx])

    def slice_sample(self, x):
        sample = []
        size = x.shape[0]
        indices = np.arange(size)
        indices = np.random.shuffle(indices)
        shuffled_indices = np.array_split(indices, size/self._batch_size)

        for index in shuffled_indices:
            sample.append(x[index])

        return sample


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', type=int, default=10000, help='batch size')
    parser.add_argument('inputfile', nargs='?', type=argparse.FileType('r'), help='data file name')
    args = parser.parse_args()

    lines = [line.strip() for line in args.inputfile.readlines()]

    np.set_printoptions(threshold=np.nan)

    db = DbscanClustering(batch_size=args.b)
    db.fit(lines)

    log_report(db.n_clusters, db.features, db.clusters)
