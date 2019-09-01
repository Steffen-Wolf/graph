import unittest
import numpy as np
import graph.nl_lmp as lmp


# TODO test with some toy problem and test for the correct solution
class TestNlLmp(unittest.TestCase):
    n_nodes = 100
    n_classes = 4
    n_edges = 500

    def get_random_problem(self):
        problem = lmp.Problem(self.n_nodes, self.n_classes)

        # set the unary costs
        for c in range(self.n_classes):
            nodes = np.arange(self.n_nodes, dtype='uint64')
            classes = np.full(self.n_nodes, c)
            costs = 2 * np.random.rand(self.n_nodes) - 1
            problem.set_unary_costs(nodes, classes, costs)

        # generate random edges
        edges = np.random.randint(0, self.n_nodes, size=(self.n_edges, 2))
        edges = edges[edges[:, 0] != edges[:, 1]]
        assert edges.max() < self.n_nodes
        # TODO get rid of duplicate edges
        class_edges = np.random.randint(0, self.n_classes, size=edges.shape)
        assert edges.shape == class_edges.shape

        # set the pair-wise cut costs
        costs = 2 * np.random.rand(len(edges)) - 1
        problem.set_pairwise_cut_costs(edges, class_edges, costs)

        # set the pair-wise join costs
        costs = 2 * np.random.rand(len(edges)) - 1
        problem.set_pairwise_join_costs(edges, class_edges, costs)

        res = lmp.Solution(self.n_nodes)
        return problem, res

    def check_res(self, res):
        partition = res.get_node_partition()
        labeling = res.get_node_labeling()
        self.assertEqual(len(partition), self.n_nodes)
        self.assertEqual(len(labeling), self.n_nodes)
        self.assertFalse(np.allclose(partition, 0))
        self.assertFalse(np.allclose(labeling, 0))
        self.assertLessEqual(labeling.max(), self.n_classes)

    def test_solve_join(self):
        problem, res = self.get_random_problem()
        res = lmp.solve_joint(problem, res)
        self.check_res(res)

    def test_solve_alternating(self):
        problem, res = self.get_random_problem()
        res = lmp.solve_alternating(problem, res)
        self.check_res(res)


if __name__ == '__main__':
    unittest.main()
