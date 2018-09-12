#include <pybind11/pybind11.h>
#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pytensor.hpp"

#include "andres/graph/graph.hxx"

namespace py = pybind11;


namespace andres {
namespace graph {

    // TODO this is just a minimal interface
    // for the graph class
    void export_graph(py::module & module){
        py::class_<Graph<>>(module, "Graph")
            .def(py::init<const std::size_t>())
            .def("insert_edge", &Graph<>::insertEdge)
            .def("insert_edges", [](Graph<> & g, const xt::pytensor<std::size_t, 2> & edges){
                py::gil_scoped_release allow_threads;
                for(size_t edge_id = 0; edge_id < edges.shape()[0]; ++edge_id) {
                    g.insertEdge(edges(edge_id, 0), edges(edge_id, 1));
                }
            })
        ;
        // TODO define properties for number of edges, number of nodes etc.
    }

}
}


PYBIND11_MODULE(_graph, module) {

    xt::import_numpy();
    module.doc() = "python-bindings for the andres graph library";

    using namespace andres::graph;
    export_graph(module);
}
