import argparse
import sys
from urllib.parse import urlparse

from urlclustering.toolkit.storage import Storage


def process_url(inputfile):
    samples_dict = {}
    total = 0
    for url in inputfile:
        line = url = url.strip()
        if line[0] == '/':
            line = line[1:]

        for word in urlparse(line).path.split("/"):
            if word in samples_dict:
                samples_dict[word][1] += 1
            else:
                samples_dict[word] = [url, 1]

        total += 1

    samples = []
    for word in samples_dict.keys():
        value = samples_dict[word]
        samples.append([word, value[0], value[1]/total, value[1]])

    Storage().batch_write_raw_data(samples)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('inputfile', nargs='?', type=argparse.FileType('r'),
                        default=sys.stdin, help='input file name')

    args = parser.parse_args()

    try:
        process_url(args.inputfile)
    finally:
        args.inputfile.close()
