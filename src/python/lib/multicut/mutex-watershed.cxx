#include <pybind11/pybind11.h>
#include "xtensor-python/pytensor.hpp"

#include "andres/graph/graph.hxx"
#include "andres/graph/multicut/mutex-watershed.hxx"

namespace py = pybind11;


namespace andres {
namespace graph {
namespace multicut {

    void export_mutex_watershed(py::module & module) {
        module.def("mutex_watershed", [](const Graph<> & graph,
                                         const xt::pytensor<double, 1> & edge_values) {
            // TODO hacked in node labels for convinience
            // const int64_t n_edges = graph.numberOfEdges();
            const int64_t n_edges = graph.numberOfVertices();
            // xt::pytensor<char, 1> edge_labels = xt::zeros<char>({n_edges});
            xt::pytensor<uint32_t, 1> edge_labels = xt::zeros<char>({n_edges});
            py::gil_scoped_release allow_threads;
            {
                mutexWatershed(graph, edge_values, edge_labels);
            }
            return edge_labels;
        }, py::arg("graph"), py::arg("edge_values"));
    }
}
}
}
