import os
import random

def log_report(n_cluster, feature_names, clusters, report_dir="./reports"):
    if not os.path.exists(report_dir):
        os.mkdir(report_dir)

    with open(f'{report_dir}/summary', "w") as summary:
        print(f'number of cluster: {n_cluster}', file=summary)

        summary.write("features\n")
        summary.write("===========================================\n")
        print(" ".join(feature_names), file=summary)

        summary.write("clusters\n")
        summary.write("===========================================\n")
        for group, cluster in clusters.items():
            sample = 'None' if len(cluster) == 0 else random.choice(cluster)
            print(f'cluster {group} ({len(cluster)})\t{sample}', file=summary)
            print()

    with open(f'{report_dir}/clusters', "w") as detail:
        for idx, cluster in clusters.items():
            print(f'\ncluster {idx}, amount: {len(cluster)}', file=detail)
            print("==================================================================", file=detail)
            print("\n".join(cluster), file=detail)
            print("==================================================================", file=detail)
