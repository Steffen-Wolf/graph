#include <pybind11/pybind11.h>
#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pytensor.hpp"

namespace py = pybind11;


namespace andres {
namespace graph {
namespace multicut {
    void export_greedy_fixation(py::module &);
    void export_mutex_watershed(py::module &);
}
}
}


PYBIND11_MODULE(_multicut, module) {

    xt::import_numpy();
    module.doc() = "multicut module of andres graph library";

    using namespace andres::graph::multicut;
    export_greedy_fixation(module);
    export_mutex_watershed(module);
}
