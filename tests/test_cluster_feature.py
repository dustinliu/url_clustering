import numpy as np
import pytest

from urlclustering.cluster_feature import UrlclusterFeature


@pytest.mark.parametrize("urls_input,expected", [
    # (["/v1/account/point"],
    #  np.array([[1, 1, 1]])),
    # (['/v1/account/point',
    #   '/v1/account/ecid'],
    #  np.array([[1, 1, 1, 0],
    #            [1, 1, 0, 1]])),
    (['/v1/account/point',
      '/v1/account/ecid',
      '/v2/test/cuur'],
     np.array([[3, 6, 3, 0, 0, 0, 0],
               [3, 6, 0, 2, 0, 0, 0],
               [0, 0, 0, 0, 1.5, 3, 2]]))
])


def test_fit(urls_input, expected):
    counter = UrlclusterFeature()
    X = counter.fit(urls_input)
    assert np.array_equal(X.toarray(), expected)

def test_word_score():
    counter = UrlclusterFeature()
    assert counter._word_score('v1', 2) == 3
    assert counter._word_score('account', 2) == 6
    assert counter._word_score('point', 1) == 3
    assert counter._word_score('v2', 1) == 1.5
    assert counter._word_score('ecid', 1) == 2
