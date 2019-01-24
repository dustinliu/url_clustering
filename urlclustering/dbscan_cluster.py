import argparse
import concurrent.futures
import os
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn import cluster
import scipy.sparse as sp

from urlclustering.cluster_feature import FeatureCounter
from urlclustering.util import elapsed_time, ending_print, log_report, read_urls


class DbscanClustering:
    def __init__(self, batch_size=10000):
        self._batch_size = batch_size
        self.n_clusters = None
        self.labels = None
        self.clusters = defaultdict(list)
        self.features = None
        self._iterations = 0

    @staticmethod
    def _fit(dataSet):
        db_ = cluster.DBSCAN(eps=50, min_samples=2, n_jobs=1).fit(dataSet)
        # return n_clusters and labels
        return len(set(db_.labels_)) - (1 if -1 in db_.labels_ else 0), db_.labels_


    def _split_fit(self, dataSet):
        self._iterations, samples = self.slice_sample(dataSet)
        n_jobs = os.cpu_count() if self._iterations > os.cpu_count() else self._iterations
        print('begin dbscan.....')
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
            iter_ = 0
            futures_ = {}
            print(f'iteration {iter_} of {self._iterations} done', end='\r')
            for sample in samples:
                futures_[executor.submit(self._fit, sp.vstack(sample['feature'].tolist()))] = sample

            for future_ in concurrent.futures.as_completed(futures_):
                sample = futures_[future_]
                sample.n_clusters, labels = future_.result()
                sample['label'] = pd.Series(labels)

                iter_ += 1
                print(f'iteration {iter_} of {self._iterations} done', end='\r')

    def fit(self, dataSet):
        start_time = time.time()
        print("collecting data...")
        feature_counter = FeatureCounter()
        dataSet = feature_counter.fit(dataSet)
        self.features = feature_counter.words

        # self.n_clusters, self.labels = self._fit(x)
        self._split_fit(dataSet)
        ending_print(f'total time spent: {elapsed_time(time.time() - start_time)}')

        # for idx, label in enumerate(self.labels):
        #     self.clusters[label].append(urls[idx])

    def slice_sample(self, dataSet):
        print('spliting data....')
        samples = []
        size = dataSet.shape[0]
        indices = np.arange(size)
        np.random.shuffle(indices)
        iterations = int(size/self._batch_size)
        shuffled_indices = np.array_split(indices, iterations)

        for index in shuffled_indices:
            samples.append(dataSet.iloc[index.tolist()].reset_index(drop=True))

        return iterations, samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', type=int, default=10000, help='batch size')
    parser.add_argument('inputfile', nargs='?', type=argparse.FileType('r'), help='data file name')
    args = parser.parse_args()

    dataSet = read_urls(args.inputfile)
    db = DbscanClustering(batch_size=args.b)
    db.fit(dataSet)

    log_report(db.n_clusters, db.features, db.clusters)
