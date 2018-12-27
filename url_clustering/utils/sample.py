import argparse
import sys

from url_clustering.noise_word import digit_ratio


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', dest='ratio', help='filter digital word')
    parser.add_argument('-l', dest='length', help='filter long word')
    parser.add_argument('sample_file', help='sample file name')

    args = parser.parse_args()
    if not (args.ratio or args.length):
        parser.error('please specify -d or -l')

    with open(sys.argv[1], 'r') as sample_file:
        for line in sample_file:
            line = line.strip()
            if len(line) > 10:
                print(line)
