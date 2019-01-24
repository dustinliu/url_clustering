import numpy as np
import pytest
import pandas as pd
import scipy.sparse as sp

from urlclustering.cluster_feature import FeatureCounter, _word_score


def test_word_score():
    assert _word_score('v1', 2) == 1
    assert _word_score('account', 2) == 4
    assert _word_score('point', 1) == 2
    assert _word_score('v2', 1) == 1
    assert _word_score('ecid', 1) == 1


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
     [sp.csr_matrix([1, 4, 2, 0, 0, 0, 0, 3 * 3]),
      sp.csr_matrix([1, 4, 0, 1, 0, 0, 0, 3 * 3]),
      sp.csr_matrix([0, 0, 0, 0, 1, 2, 1, 3 * 3])])
])
def test_fit(urls_input, expected):
    dataSet = pd.DataFrame(urls_input, columns=['url'])
    counter = FeatureCounter()
    counter.fit(dataSet)
    assert dataSet['feature'].tolist() == expected
