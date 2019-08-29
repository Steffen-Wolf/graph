#include <pybind11/pybind11.h>
#include "xtensor-python/pytensor.hpp"

#include "nl-lmp/solve-joint.hxx"
#include "andres/graph/graph.hxx"

namespace py = pybind11;


namespace nl_lmp {

    void export_solve_joint(py::module & module) {
        typedef andres::graph::Graph<> Graph;
        module.def("solve_joint", [](const xt::pytensor<std::size_t, 2> & uv_ids,
                                     const xt::pytensor<double, 1> & edge_values) {
            py::gil_scoped_release allow_threads;
            Solution input(1);
            Problem<Graph> problem(1, 1);
            {
                const Solution result = update_labels_and_multicut(problem, input);
            }
        }, py::arg("uv-ids"), py::arg("edge_values"));
    }
}
