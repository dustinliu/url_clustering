import argparse
import sys

from urlclustering.noise_word import digit_ratio


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', dest='ratio', help='filter digital word')
    parser.add_argument('-l', dest='length', help='filter long word')
    parser.add_argument('sample_file', nargs='?', type=argparse.FileType('r'),
                        default=sys.stdin, help='sample file name')
    args = parser.parse_args()

    if args.ratio:
        filter = lambda x: digit_ratio(x) < float(args.ratio)
    elif args.length:
        filter = lambda x: len(x) > int(args.length)
    else:
        filter = lambda _: True


    for line in args.sample_file:
        line = line.strip()
        if filter(line):
            print(line)
