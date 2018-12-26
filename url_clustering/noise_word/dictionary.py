import os
import sys
import pickle


class Singleton(type):
    _instance = None

    def __call__(self, *args, **kwargs):
        if not self._instance :
            self._instance = super(Singleton, self).__call__(*args, **kwargs)
        return self._instance


class Dictionary(metaclass=Singleton):
    idx_set = set()

    def __init__(self):
        with open(f'{os.path.dirname(__file__)}/dict.set', 'rb') as dict_file:
            self.idx_set = pickle.load(dict_file)
            print(len(self.idx_set))

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
                if not " " in word:
                    words.add(word)

                idx_file.read(8)

        dest_dir = os.path.dirname(__file__)
        if not dest_dir:
            dest_dir = '.'
        with open(f'{dest_dir}/dict.set', 'wb') as dict_file:
            pickle.dump(words, dict_file)
            print(f'imported {len(words)} words')

    def __contains__(self, item):
        return item in self.idx_set


if __name__ == '__main__':
    Dictionary.read_star_dict(sys.argv[1])
