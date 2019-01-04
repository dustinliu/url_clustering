import os
import random

def log_report(n_cluster, feature_names, clusters, report_dir="./reports"):
    if not os.path.exists(report_dir):
        os.mkdir(report_dir)

    with open(f'{report_dir}/summary', "w") as summary:
        summary.write(f'number of cluster: {n_cluster}\n\n')

        summary.write("features\n")
        summary.write("===========================================\n")
        for word in feature_names:
            summary.write(f'{word}\n')
        summary.write("\n")

        summary.write("clusters\n")
        summary.write("===========================================\n")
        for i in range(len(clusters)):
            summary.write(f'cluster {i} ({len(clusters[i])})\t{random.choice(clusters[i])}')
            summary.write("\n")

    with open(f'{report_dir}/clusters', "w") as detail:
        for i in range(len(clusters)):
            print(f'\ngroup {str(i)}, amount: {str(len(clusters[i]))}', file=detail)
            print("=====================================================================================", file=detail)
            print("\n".join(clusters[i]), file=detail)
            print("=====================================================================================", file=detail)
