from urllib.parse import urlparse


def tokenize_url(url):
    path = urlparse(url).path
    if path[0] == '/':
        path = path[1:]

    return [word.lower() for word in path.split('/')]
