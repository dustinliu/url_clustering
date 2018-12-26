import numpy as np

from url_clustering.noise_word.dictionary import Dictionary


class NoiseWordDetector:
    dict_ = Dictionary()

    def __init__(self, max_len=200, min_len=0):
        self.max_len = max_len
        self.min_len = min_len

    def is_noise_word(self, word):
        if not self.min_len < len(word) < self.max_len:
            return True

        sample = word.lower()
        if sample in self.dict_:
            return False

        return True

    def _extract_features(self, words):
        features = []
        for word in [w.lower() for w in words]:
            features.append(np.array([len(word), self._digit_ratio(word), self._readability(word)]))

        return features

    @staticmethod
    def _digit_ratio(word):
        if len(word) == 0:
            return 0

        digit_count = 0
        for c in word:
            if '0' <= c <= '9':
                digit_count += 1
        return digit_count / len(word)

    def _readability(self, word):
        return sum([len(token) for token in self._tokenize(word)]) / len(word)

    def _tokenize(self, word):
        tokens = []
        word_len = len(word)
        i = 0
        while i < word_len - 3:
            token_len = word_len - i
            pos = 45 if token_len > 45 else token_len  # the length of the longest english word is 45
            for j in range(pos, 2, -1):
                token = word[i:i+j]
                if token in self.dict_:
                    tokens.append(token)
                    i += len(token) - 1
                    break
            i += 1
        return tokens


if __name__ == '__main__':
    pass
