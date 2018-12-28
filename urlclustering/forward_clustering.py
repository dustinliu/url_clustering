import sys

from urlclustering.feature import tokenize
from urlclustering.report import log_report


class ForwardClustering:
    def __init__(self):
        self.clusters = None
        self.feature_names = None

    def fit(self, urls):
        features = [tokenize(url, True) for url in urls]
        self.feature_names = list(set([word for f in features for word in f]))

        clusters_dict = dict()
        for idx, feature in enumerate(features):
            if repr(feature) in clusters_dict:
                clusters_dict[repr(feature)].append(feature)
            else:
                clusters_dict[repr(feature)] = [feature]

        print(clusters_dict)
        self.clusters = clusters_dict.values()
        log_report(len(self.clusters), self.feature_names, self.clusters)


if __name__ == "__main__":
    with open(sys.argv[1]) as fp:
        lines = [line.rstrip() for line in fp.readlines()]

    ForwardClustering().fit(lines)
