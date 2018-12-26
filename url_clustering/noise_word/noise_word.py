from url_clustering.noise_word.dictionary import Dictionary
import numpy as np


class NoiseWordDetector:
    dict_ = Dictionary()

    def __init__(self):
        pass

    def is_noise_word(self, word):
        if word in self.dict_:
            return False
        return True

    def _digit_ratio(self, word):
        if len(word) == 0:
            return 0

        digit_count = 0
        for c in word:
            if '0' <= c <= '9':
                digit_count += 1
        return digit_count / len(word)

    def _readability(self, word):
        if word in self.dict_:
            return 1

    def _extract_features(self, words):
        features = []
        for word in [word.lower() for word in words]:
            features.append(np.array([len(word), self._digit_ratio(word), self._readability(word)]))

        return features


if __name__ == '__main__':
    pass
