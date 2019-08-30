#include <pybind11/pybind11.h>
#include "xtensor-python/pytensor.hpp"

#include "nl-lmp/greedy-additive.hxx"
#include "andres/graph/graph.hxx"

namespace py = pybind11;


namespace nl_lmp {
    void export_greedy_additive(py::module & module) {
        typedef andres::graph::Graph<> Graph;
        module.def("greedy_additive", [](const Problem<Graph> & problem, const Solution & input_solution) {
            Solution result(input_solution.size());
            {
                py::gil_scoped_release allow_threads;
                result = greedyAdditiveEdgeContraction(problem, input_solution);
            }
            return result;
        }, py::arg("problem"), py::arg("initial_solution"));
    }
}
