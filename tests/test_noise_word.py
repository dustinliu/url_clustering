import pytest

from tests.test_utils import compare_list
from urlclustering.noise_feature import pos, tokenize, readability, digit_ratio, special_char_ratio


class TestNoiseWord:
    def test_digit_ratio(self):

        assert digit_ratio("a1b0") == 0.5
        assert digit_ratio('ab75685794') == 0.8
        assert digit_ratio('92') == 1

    def test_pos(self):
        assert pos('v1', '/v1/test/get') == 0

    def test_tokenize(self):
        assert compare_list(tokenize("paymentuser"), ['payment', 'user'])
        assert compare_list(tokenize("epaymentusere"), ['payment', 'user'])
        assert compare_list(tokenize("epaymentusere"), ['payment', 'user'])
        assert compare_list(tokenize("epaymenteusere"), ['payment', 'user'])
        assert compare_list(tokenize("eepaymenteeuseree"), ['payment', 'user'])
        assert compare_list(tokenize("paymentgateway"), ['payment', 'gateway'])
        assert compare_list(tokenize("paymentgateway"), ['payment', 'gateway'])

    def test_readability(self):
        assert readability("env.default") == pytest.approx(0.909090909090)
        assert readability(".env") == pytest.approx(0.75)
        assert readability("api") == pytest.approx(1)

    def test_special_char_ratio(self):
        assert special_char_ratio("a%") == pytest.approx(0.5)
        assert special_char_ratio("z_") == pytest.approx(0)
        assert special_char_ratio("0-") == pytest.approx(0)
        assert special_char_ratio("9+") == pytest.approx(0.5)
