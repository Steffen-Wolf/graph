#include <pybind11/pybind11.h>
#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pytensor.hpp"

namespace py = pybind11;


namespace nl_lmp {
    void export_problem(py::module &);
    void export_solution(py::module &);
    void export_solve_alternating(py::module &);
    void export_solve_joint(py::module &);
    void export_greedy_additive(py::module &);
}


PYBIND11_MODULE(_nl_lmp, module) {

    xt::import_numpy();
    module.doc() = "nl-lmp module of andres graph library";

    using namespace nl_lmp;
    export_problem(module);
    export_solution(module);
    export_solve_alternating(module);
    export_solve_joint(module);
    export_greedy_additive(module);
}
