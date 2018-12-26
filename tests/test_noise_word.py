import pytest

from url_clustering.noise_word.noise_word import _digit_ratio, is_noise_word


class TestNoiseWord:
    def test_digit_ratio(self):
        assert _digit_ratio("a1b0") == 0.5
        print(_digit_ratio('ab75685794'))
        assert _digit_ratio('ab75685794') == 0.8

    def test_is_noise_word(self):
        assert is_noise_word("user") == False

