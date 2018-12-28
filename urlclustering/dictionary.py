import os
import pickle
import re

import argparse


class Dictionary:
    _idx_set = set()
    _idx_file_name = f'{os.path.dirname(__file__)}/dict.set'
    _ies_re = re.compile('ies$')

    def __init__(self):
        with open(self._idx_file_name, 'rb') as dict_file:
            self._idx_set = pickle.load(dict_file)

    def add(self, *words):
        self._idx_set.update(words)

    def save(self):
        with open(self._idx_file_name, 'wb') as idx_file:
            pickle.dump(self._idx_set, idx_file)

    @staticmethod
    def read_star_dict(idx_file_location):
        words = set()
        with open(idx_file_location, 'rb') as idx_file:
            while True:
                word_bytes = bytearray()
                one_byte = idx_file.read(1)

                if one_byte == b'':
                    break

                while ord(one_byte) != 0:
                    word_bytes.append(ord(one_byte))
                    one_byte = idx_file.read(1)

                word = word_bytes.decode()
                if " " not in word:
                    words.add(word)

                idx_file.read(8)

        return words

    def __contains__(self, word):
        found = word in self._idx_set
        if not found and word.endswith("ies"):
            found = re.sub(self._ies_re, 'y', word) in self._idx_set

        return found


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--add', dest='add', default=False, action='store_true', help='add word(s) to dictionary')
    parser.add_argument('word', nargs='+')
    args = parser.parse_args()

    dict_ = Dictionary()
    if args.add:
        dict_.add(*args.word)
        dict_.save()
    else:
        for word in args.word:
            print(f'{word} : {word in dict_}')
