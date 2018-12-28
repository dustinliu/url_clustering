import argparse
import csv
import sys
from urllib.parse import urlparse


def process_url(inputfile):
    result = {}
    total = 0
    for url in inputfile:
        line = url = url.strip()
        if line[0] == '/':
            line = line[1:]

        for word in urlparse(line).path.split("/"):
            if word in result:
                result[word][1] += 1
            else:
                result[word] = [url, 1]

        total += 1

    writer = csv.writer(sys.stdout, delimiter='\t')
    writer.writerow(['word', 'url', 'frequency', 'amount'])
    for word in result.keys():
        writer.writerow([word, result[word][0], result[word][1]/total, result[word][1]])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('inputfile', nargs='?', type=argparse.FileType('r'),
                        default=sys.stdin, help='input file name')

    args = parser.parse_args()

    try:
        process_url(args.inputfile)
    finally:
        args.inputfile.close()
