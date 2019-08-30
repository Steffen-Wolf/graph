#include <pybind11/pybind11.h>
#include "xtensor-python/pytensor.hpp"
#include "nl-lmp/solution.hxx"

namespace py = pybind11;


namespace nl_lmp {

    void export_solution(py::module & module) {

        typedef xt::pytensor<size_t, 1> NodeVector;

        py::class_<Solution>(module, "Solution")
            .def(py::init<size_t>())
            //
            .def("get_node_partition", [](const Solution & self){
                NodeVector partition = xt::zeros<size_t>({self.size()});
                for(size_t i = 0; i < self.size(); ++i) {
                    partition(i) = self[i].clusterIndex;
                }
                return partition;
            })
            //
            .def("get_node_labeling", [](const Solution & self){
                NodeVector labeling = xt::zeros<size_t>({self.size()});
                for(size_t i = 0; i < self.size(); ++i) {
                    labeling(i) = self[i].classIndex;
                }
                return labeling;
            })
        ;
    }
}
