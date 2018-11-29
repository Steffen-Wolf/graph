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
            KernighanLinSettings setting;
            setting.epsilon = epsilon;

            // TODO hacked in node labels for convinience
            const int64_t n_edges = graph.numberOfVertices();
            
            // warmstart KL with one cluster per node
            std::vector<uint64_t> edge_labels = std::vector<uint64_t>(n_edges);
            int k = 0;
            std::fill(edge_labels.begin(), edge_labels.end(), k++);

            xt::pytensor<uint64_t, 1> edge_labels_exp = xt::zeros<char>({n_edges});
            py::gil_scoped_release allow_threads;
            {
                std::vector<uint64_t> tmp_labels = kernighanLin(graph, edge_values, edge_labels, setting);
                // convert std::vector edge_labels to xtensor array for export
                for (int i = 0; i < n_edges; ++i){
                    edge_labels_exp(i) = tmp_labels[i];
                }
            }

            return edge_labels_exp;
        }, py::arg("graph"), py::arg("edge_values"), py::arg("epsilon"));
    }
}
}
}
