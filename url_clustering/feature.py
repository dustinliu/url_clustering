from urllib.parse import urlparse
import re

digit_set = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}
digit_white_list = [re.compile(r'^v\d+')]


def tokenize(url, noise_canceling=True):
    paths = urlparse(url).path.split("/")
    len_ = len(paths)
    if noise_canceling:
        return len_, [word for word in paths if not is_noise_word(word)]

    return len_, paths

def is_noise_word(word):
    if len(word) == 0:
        return False

    return is_digital_word(word)

def is_digital_word(word, threshold=0.5):
    for pattern in digit_white_list:
        if pattern.match(word):
            return False

    if len(word) == 0:
        return False

    digit_count = 0
    for c in word:
        if c in digit_set:
            digit_count += 1

    return digit_count / len(word) >= threshold
