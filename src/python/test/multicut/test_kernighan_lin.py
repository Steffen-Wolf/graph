import unittest
import numpy as np


# minimal test for the graph class
class TestKernighanLin(unittest.TestCase):
    def test_kernighan_lin(self):
        import graph as ag
        import graph.multicut as mc

        # a simple weighted graph in which an optimal multicut is non-trivial
        g = ag.Graph(6)
        uv_ids = np.array([[0, 1], # 0
                           [0, 3], # 1
                           [1, 2], # 2
                           [1, 4], # 3
                           [2, 5], # 4
                           [3, 4], # 5
                           [4, 5]]) # 6

        weights = np.array([5, -20, 5, 5, -20, 5, 5],
                           dtype='float64')
        # in the c++ unit test, all edges are zero = all nodes connected
        # in initial solution
        labels = mc.kernighan_lin(g, weights, 1e-6)
        self.assertEqual(len(labels), 6)

        unique_res = np.unique(labels)
        self.assertEqual(len(unique_res), 2)
        self.assertEqual(labels[0], labels[3])
        self.assertEqual(labels[0], labels[4])
        self.assertEqual(labels[0], labels[5])
        self.assertEqual(labels[1], labels[2])


if __name__ == '__main__':
    unittest.main()
