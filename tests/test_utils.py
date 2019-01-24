from urlclustering.util import split_every, execute_fork_join

def s(l):
    return [d ** 2 for d in l]

class TestUtils:
    def test_split_every(self):
        a = range(10)
        assert list(split_every(3, a)) ==  [[0, 1, 2], [3, 4, 5], [6, 7 ,8], [9]]

    def test_execute_fork_join(self):

        a = range(10)
        assert execute_fork_join(s, a, batch_size=3, max_workers=2) == [x ** 2 for x in a]
