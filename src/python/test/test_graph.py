import unittest
import numpy as np


# minimal test for the graph class
class TestGraph(unittest.TestCase):
    def test_graph(self):
        import graph as ag
        g = ag.Graph(5)

        g.insert_edge(0, 1)
        g.insert_edge(1, 2)

        edges = np.array([[2, 3], [3, 4]], dtype='uint64')
        g.insert_edges(edges)

        # TODO export number of vertices and edges for graph and check that
        # we get correct values


if __name__ == '__main__':
    unittest.main()
