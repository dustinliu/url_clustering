from urlclustering.dictionary import Dictionary
from urlclustering.utils import tokenize_url

dict_ = Dictionary()


def pos(word, url):
    tokens = tokenize_url(url)
    return tokens.index(word) / len(tokens)


def tokenize(word):
    tokens = []
    word_len = len(word)
    i = 0
    while i <= word_len - 3:
        token_len = word_len - i
        pos_ = 45 if token_len > 45 else token_len  # the length of the longest english word is 45
        for j in range(pos_, 2, -1):
            token = word[i:i+j]
            if token in dict_:
                tokens.append(token)
                i += len(token) - 1
                break
        i += 1
    return tokens


def readability(word):
    if len(word) == 0:
        return 1

    len_ = 0
    for token in tokenize(word):
        len_ += len(token)
    return len_ / len(word)


def digit_ratio(word):
    try:
        if word is None or len(word) == 0:
            return 0
    except TypeError:
        print(f'type error: {word}, type: {type(word)}')
        return 0

    digit_count = 0
    for c in word:
        if '0' <= c <= '9':
            digit_count += 1
    return digit_count / len(word)


def extract_features(word, url):
    word = word.lower()
    return pos(word, url), len(word), readability(word), digit_ratio(word)
