from urllib.parse import urlparse

from urlclustering.dictionary import Dictionary

dict_ = Dictionary()


def urls_preprocess(urls):
    return list(set([urlparse(u).path for u in urls]))

def tokenize_url(url, max_digital_ratio=1):
    path = urlparse(url).path
    if path[0] == '/':
        path = path[1:]
    if path[-1] == '/':
        path = path[:-1]

    return [word for word in path.split('/') if digit_ratio(word) <= max_digital_ratio]

def pos(word, url):
    tokens = tokenize_url(url)
    p = tokens.index(word)
    len_ = len(tokens) - 1
    return 0 if len_ == 0 and p == 0 else tokens.index(word) / (len(tokens) - 1)


def tokenize(word):
    tokens = []
    word_len = len(word)
    i = 0
    while i <= word_len - 3:
        token_len = word_len - i

        # the length of the longest english word is 45
        pos_ = 45 if token_len > 45 else token_len
        for j in range(pos_, 2, -1):
            token = word[i:(i + j)]
            if token in dict_:
                tokens.append(token)
                i += len(token) - 1
                break
        i += 1
    return tokens


def special_char_ratio(word):
    if word is None or len(word) == 0:
        return 0

    len_ = 0
    for c in word.lower():
        if not 'a' <= c <= 'z' and not '0' <= c <= '9'\
           and c != '_' and c != '-':
            len_ += 1

    return len_ / len(word)


def readability(word):
    if len(word) == 0:
        return 1

    len_ = 0
    for token in tokenize(word):
        len_ += len(token)
    return len_ / len(word)


def digit_ratio(word):
    if word is None or len(word) == 0:
        return 0

    digit_count = 0
    for c in word:
        if '0' <= c <= '9':
            digit_count += 1
    return digit_count / len(word)


def extract_features(word, url):
    word = word.lower()
    return pos(word, url), len(word), readability(word), digit_ratio(word), special_char_ratio(word)
