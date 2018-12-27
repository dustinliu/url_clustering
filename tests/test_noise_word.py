from tests.utils import compare_list
from url_clustering.noise_word import NoiseWordDetector


class TestNoiseWord:
    def test_digit_ratio(self):
        assert NoiseWordDetector._digit_ratio("a1b0") == 0.5
        assert NoiseWordDetector._digit_ratio('ab75685794') == 0.8

    def test_is_noise_word(self):
        assert NoiseWordDetector().is_noise_word("user") == False
        assert NoiseWordDetector(min_len=10).is_noise_word("user") == True
        assert NoiseWordDetector(max_len=3).is_noise_word("user") == True

    def test_tokenize(self):
        assert compare_list(NoiseWordDetector()._tokenize("paymentuser"), ['payment', 'user'])
        assert compare_list(NoiseWordDetector()._tokenize("epaymentusere"), ['payment', 'user'])
        assert compare_list(NoiseWordDetector()._tokenize("epaymentusere"), ['payment', 'user'])
        assert compare_list(NoiseWordDetector()._tokenize("epaymenteusere"), ['payment', 'user'])
        assert compare_list(NoiseWordDetector()._tokenize("eepaymenteeuseree"), ['payment', 'user'])
        assert compare_list(NoiseWordDetector()._tokenize("paymentgateway"), ['payment', 'gateway'])
        assert compare_list(NoiseWordDetector()._tokenize("paymentgateway"), ['payment', 'gateway'])

    def test_readability(self):
        assert NoiseWordDetector()._readability("payment") == 1
        assert NoiseWordDetector()._readability("paymentdlaoiek") == 0.5
        assert NoiseWordDetector()._readability("paymentdlaoieklaqeuser") == 0.5

