import argparse
from collections import defaultdict

import numpy as np
from sklearn import cluster

from urlclustering.cluster_feature import FeatureCounter
from urlclustering.report import log_report


class DbscanClustering:
    def __init__(self):
        self.n_clusters = None
        self.labels = None
        self.clusters = defaultdict(list)
        self.features = None

    def fit(self, urls):
        print("collecting data...")
        feature_counter = FeatureCounter()
        sample_urls, x = feature_counter.fit(urls)
        self.x = x
        self.features = feature_counter.words

        # print('sampling data...')
        # x = self.data_sampling(x)

        print("running dbscan...")
        db = cluster.DBSCAN(eps=50, min_samples=2).fit(x)
        self.labels = db.labels_
        self.n_clusters = len(set(self.labels)) - (1 if -1 in self.labels else 0)

        for idx, label in enumerate(self.labels):
            self.clusters[label].append(sample_urls[idx])

    def data_sampling(self, x):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('inputfile', nargs='?', type=argparse.FileType('r'), help='data file name')
    args = parser.parse_args()

    lines = [line.strip() for line in args.inputfile.readlines()]

    np.set_printoptions(threshold=np.nan)

    db = DbscanClustering()
    db.fit(lines)

    log_report(db.n_clusters, db.features, db.clusters)
