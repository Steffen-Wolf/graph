#include <pybind11/pybind11.h>
#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pytensor.hpp"

namespace py = pybind11;


namespace nl_lmp {
    void export_solve_joint(py::module &);
}


PYBIND11_MODULE(_nl_lmp, module) {

    xt::import_numpy();
    module.doc() = "nl-lmp module of andres graph library";

    using namespace nl_lmp;
    export_solve_joint(module);
}
