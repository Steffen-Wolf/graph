#include <pybind11/pybind11.h>
#include "xtensor-python/pytensor.hpp"

#include "andres/graph/graph.hxx"
#include "andres/graph/multicut/greedy-fixation.hxx"

namespace py = pybind11;


namespace andres {
namespace graph {
namespace multicut {

    void export_greedy_fixation(py::module & module) {
        module.def("greedy_fixation", [](const Graph<> & graph,
                                         const xt::pytensor<double, 1> & edge_values) {
            // TODO should we do a consistency check that number 
            // of edges in graph and edge values agree ?
            const int64_t n_edges = graph.numberOfEdges();
            xt::pytensor<char, 1> edge_labels = xt::zeros<char>({n_edges});
            py::gil_scoped_release allow_threads;
            {
                greedyFixation(graph, edge_values, edge_labels);
            }
            return edge_labels;
        }, py::arg("graph"), py::arg("edge_values"));
    }
}
}
}
