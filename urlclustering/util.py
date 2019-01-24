import concurrent.futures
import csv
import datetime
import math
import os
import random
import sys
from itertools import islice
import pandas as pd

terminal_row = 48
terminal_columns = 160
if sys.stdout.isatty():
    terminal_row, terminal_columns = os.popen('stty size', 'r').read().split()
    terminal_columns = int(terminal_columns)
    terminal_row = int(terminal_row)


def read_urls(fh):
    return pd.read_csv(fh, index_col=False, header=None, names=['url'], sep='\s+')

def log_report(n_cluster, feature_names, clusters, report_dir="./reports"):
    if not os.path.exists(report_dir):
        os.mkdir(report_dir)

    with open(f'{report_dir}/summary', "w") as summary:
        print(f'number of cluster: {n_cluster}', file=summary)

        print("features", file=summary)
        print("===========================================", file=summary)
        print(" ".join(feature_names), file=summary)

        print("clusters", file=summary)
        print("===========================================", file=summary)
        for group, cluster in clusters.items():
            sample = 'None' if len(cluster) == 0 else random.choice(cluster)
            print(f'cluster {group:<3} {len(cluster):>6} {sample}', file=summary)

    with open(f'{report_dir}/clusters', "w") as detail:
        for idx, cluster in clusters.items():
            print(f'\ncluster {idx}, amount: {len(cluster)}', file=detail)
            print("==================================================================", file=detail)
            print("\n".join(cluster), file=detail)
            print("==================================================================", file=detail)


def ending_print(message):
    print(' ' * terminal_columns, end='\r')
    print(message)


def elapsed_time(seconds):
    return str(datetime.timedelta(seconds=(seconds)))


def split_every(n, iterable):
    i = iter(iterable)
    piece = list(islice(i, n))
    while piece:
        yield piece
        piece = list(islice(i, n))


def execute_fork_join(fn, iterable, batch_size=10000, max_workers=None):
    batch_num = math.ceil(len(iterable)/batch_size)
    if not max_workers:
        max_workers = os.cpu_count() if batch_num > os.cpu_count() else batch_num

    result = []
    chunked_data = list(split_every(batch_size, iterable))
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures_ = [executor.submit(fn, data) for data in chunked_data]
        for future_ in futures_:
            result.extend(future_.result())
    return result
