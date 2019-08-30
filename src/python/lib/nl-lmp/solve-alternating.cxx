#include <pybind11/pybind11.h>
#include "xtensor-python/pytensor.hpp"

#include "nl-lmp/solve-alternating.hxx"
#include "andres/graph/graph.hxx"

namespace py = pybind11;


namespace nl_lmp {
    void export_solve_alternating(py::module & module) {
        typedef andres::graph::Graph<> Graph;
        module.def("solve_alternating", [](const Problem<Graph> & problem, const Solution & input_solution) {
            Solution result(input_solution.size());
            {
                py::gil_scoped_release allow_threads;
                result = solve_alternating(problem, input_solution);
            }
            return result;
        }, py::arg("problem"), py::arg("initial_solution"));
    }
}
