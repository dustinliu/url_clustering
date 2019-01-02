import pytest

from tests.utils import compare_list
from urlclustering.noise_feature import pos, tokenize, readability, digit_ratio
from urlclustering.noise_word import NoiseWordDetector


class TestNoiseWord:
    def test_digit_ratio(self):
        assert digit_ratio("a1b0") == 0.5
        assert digit_ratio('ab75685794') == 0.8
        assert digit_ratio('92') == 1

    def test_is_noise_word(self):
        assert NoiseWordDetector().is_noise_word("user") == False

    def test_pos(self):
        assert pos('v1', '/v1/test/get') == 0

    def test_tokenize(self):
        assert compare_list(tokenize(NoiseWordDetector().dict_, "paymentuser"), ['payment', 'user'])
        assert compare_list(tokenize(NoiseWordDetector().dict_, "epaymentusere"), ['payment', 'user'])
        assert compare_list(tokenize(NoiseWordDetector().dict_, "epaymentusere"), ['payment', 'user'])
        assert compare_list(tokenize(NoiseWordDetector().dict_, "epaymenteusere"), ['payment', 'user'])
        assert compare_list(tokenize(NoiseWordDetector().dict_, "eepaymenteeuseree"), ['payment', 'user'])
        assert compare_list(tokenize(NoiseWordDetector().dict_, "paymentgateway"), ['payment', 'gateway'])
        assert compare_list(tokenize(NoiseWordDetector().dict_, "paymentgateway"), ['payment', 'gateway'])

    def test_readability(self):
        assert readability(NoiseWordDetector().dict_, "env.default") == pytest.approx(0.909090909090)
        assert readability(NoiseWordDetector().dict_, ".env") == pytest.approx(0.75)

