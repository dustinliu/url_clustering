from unittest import TestCase

from url_clustering.feature import is_digital_word, is_noise_word


class TestFeature(TestCase):
    def test_is_digital_word(self):
        self.assertTrue(is_digital_word("a098789"))
        self.assertFalse(is_digital_word("afdsafdsf"))
        self.assertFalse(is_digital_word("v1"))

