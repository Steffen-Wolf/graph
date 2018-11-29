#include <pybind11/pybind11.h>
#include "xtensor-python/pytensor.hpp"

#include "andres/graph/graph.hxx"
#include "andres/graph/multicut/kernighan-lin.hxx"

namespace py = pybind11;


namespace andres {
namespace graph {
namespace multicut {

    void export_kernighan_lin(py::module & module) {
        module.def("kernighan_lin", [](const Graph<> & graph,
                                         const xt::pytensor<double, 1> & edge_values,
                                         const float epsilon) {
            const KernighanLinSettings setting;
            // setting.epsilon = epsilon;

            // TODO hacked in node labels for convinience
            // const int64_t n_edges = graph.numberOfEdges();
            const int64_t n_edges = graph.numberOfVertices();
            // xt::pytensor<char, 1> edge_labels = xt::zeros<char>({n_edges});
            //xt::pytensor<uint32_t, 1> edge_labels = xt::zeros<char>({n_edges});
            std::vector<uint64_t> edge_labels = std::vector<uint64_t>(n_edges);
            py::gil_scoped_release allow_threads;
            {
                kernighanLin(graph, edge_values, edge_labels, setting);
            }

            // convert std::vector edge_labels to xtensor array for export
            xt::pytensor<uint64_t, 1> edge_labels_exp = xt::zeros<char>({n_edges});
            for (int i = 0; i < n_edges; ++i){
                edge_labels_exp(i) = edge_labels[i];
            }

            return edge_labels_exp;
        }, py::arg("graph"), py::arg("edge_values"), py::arg("epsilon"));
    }
}
}
}
