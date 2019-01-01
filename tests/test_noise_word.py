import pytest

from tests.utils import compare_list
from urlclustering.noise_word import NoiseWordDetector, digit_ratio


class TestNoiseWord:
    def test_digit_ratio(self):
        assert digit_ratio("a1b0") == 0.5
        assert digit_ratio('ab75685794') == 0.8
        assert digit_ratio('92') == 1

    def test_is_noise_word(self):
        assert NoiseWordDetector().is_noise_word("user") == False

    def test_pos(self):
        assert NoiseWordDetector()._pos('v1', '/v1/test/get') == 0

    def test_tokenize(self):
        assert compare_list(NoiseWordDetector()._tokenize("paymentuser"), ['payment', 'user'])
        assert compare_list(NoiseWordDetector()._tokenize("epaymentusere"), ['payment', 'user'])
        assert compare_list(NoiseWordDetector()._tokenize("epaymentusere"), ['payment', 'user'])
        assert compare_list(NoiseWordDetector()._tokenize("epaymenteusere"), ['payment', 'user'])
        assert compare_list(NoiseWordDetector()._tokenize("eepaymenteeuseree"), ['payment', 'user'])
        assert compare_list(NoiseWordDetector()._tokenize("paymentgateway"), ['payment', 'gateway'])
        assert compare_list(NoiseWordDetector()._tokenize("paymentgateway"), ['payment', 'gateway'])

    def test_readability(self):
        assert NoiseWordDetector()._readability("env.default") == pytest.approx(0.909090909090)
        assert NoiseWordDetector()._readability(".env") == pytest.approx(0.75)

