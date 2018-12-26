

def compare_list(actual, expect):
    return all([a == b for a, b in zip(actual, expect)])
