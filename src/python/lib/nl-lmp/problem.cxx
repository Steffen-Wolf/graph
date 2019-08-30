#include <pybind11/pybind11.h>
#include "xtensor-python/pytensor.hpp"

#include "nl-lmp/problem.hxx"
#include "andres/graph/graph.hxx"

namespace py = pybind11;


namespace nl_lmp {

    void export_problem(py::module & module) {

        typedef xt::pytensor<size_t, 1> NodeVector;
        typedef xt::pytensor<size_t, 2> EdgeVector;
        typedef xt::pytensor<double, 1> CostVector;
        typedef Problem<andres::graph::Graph<>> P;

        py::class_<P>(module, "Problem")
            .def(py::init<size_t, size_t>())
            //
            .def("set_unary_costs", [](P & self,
                                       const NodeVector & us,
                                       const NodeVector & ls,
                                       const CostVector & costs){
                py::gil_scoped_release lift_gil;
                for(size_t i = 0; i < us.size(); ++i) {
                    self.setUnaryCost(us(i), ls(i), costs(i));
                }
            }, py::arg("nodes"), py::arg("labels"), py::arg("costs"))
            //
            .def("set_pairwise_cut_costs", [](P & self,
                                              const NodeVector & uvs,
                                              const NodeVector & lms,
                                              const CostVector & costs,
                                              const bool add_edge_into_original_graph){
                py::gil_scoped_release lift_gil;
                for(size_t i = 0; i < uvs.size(); ++i) {
                    self.setPairwiseCutCost(uvs(i, 0), uvs(i, 1),
                                            lms(i, 0), lms(i, 1),
                                            costs(i), add_edge_into_original_graph);
                }
            }, py::arg("uvs"), py::arg("labels"), py::arg("costs"), py::arg("add_edge_into_original_graph")=true)
            //
            .def("set_pairwise_join_costs", [](P & self,
                                              const NodeVector & uvs,
                                              const NodeVector & lms,
                                              const CostVector & costs,
                                              const bool add_edge_into_original_graph){
                py::gil_scoped_release lift_gil;
                for(size_t i = 0; i < uvs.size(); ++i) {
                    self.setPairwiseJoinCost(uvs(i, 0), uvs(i, 1),
                                             lms(i, 0), lms(i, 1),
                                             costs(i), add_edge_into_original_graph);
                }
            }, py::arg("uvs"), py::arg("labels"), py::arg("costs"), py::arg("add_edge_into_original_graph")=true)
        ;
    }
}
