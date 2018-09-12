import unittest
import numpy as np


# minimal test for the graph class
class TestGreedyFixation(unittest.TestCase):
    # TODO test fixed toy example for correct result
    def test_greedy_fixation(self):
        import graph as ag
        import graph.multicut as mc

        g = ag.Graph(10)
        edges = np.concatenate((np.arange(9)[:, None],
                                np.arange(1, 10)[:, None]), axis=1).astype('uint64')
        g.insert_edges(edges)

        edge_values = np.random.rand(edges.shape[0])

        edge_labels = mc.greedy_fixation(g, edge_values)
        self.assertEqual(len(edge_values), len(edge_labels))


if __name__ == '__main__':
    unittest.main()
