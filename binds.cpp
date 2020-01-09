#include "conv_layer.hpp"
#include "cross_entropy.hpp"
#include "dense_layer.hpp"
#include "max_pool.hpp"
#include "mnist.hpp"
#include "relu.hpp"
#include "softmax.hpp"
#include <algorithm>
#include <eigen3/Eigen/Dense>
#include <fstream>
#include <iostream>
#include <math.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdlib.h>
#include <vector>
using namespace std;
using Eigen::MatrixXd;
namespace py = pybind11;

PYBIND11_MODULE(my_project, m) {
  py::class_<Conv_Layer>(m, "Conv_Layer")
      .def(py::init<int, int, int, int, int, int>(), "Convolutional Layer",
           py::arg("height"), py::arg("width"), py::arg("depth"),
           py::arg("filter_size"), py::arg("stride"), py::arg("num_filters"))
      .def("feed_forward", &Conv_Layer::feed_forward)
      .def("back_propagation", &Conv_Layer::back_propagation)
      .def("update_weights", &Conv_Layer::update_weights)
      .def("save_filters", &Conv_Layer::save_filters)
      .def("load_filters", &Conv_Layer::load_filters)
      .def_readwrite("gradients", &Conv_Layer::gradients)
      .def_readwrite("filters", &Conv_Layer::filters)
      .def_readwrite("output", &Conv_Layer::output);
  py::class_<Cross_Entropy>(m, "Cross_Entropy")
      .def(py::init<int>(), "Cross Entropy Layer")
      .def("feed_forward", &Cross_Entropy::feed_forward)
      .def("back_propagation", &Cross_Entropy::back_propagation)
      .def_readwrite("loss", &Cross_Entropy::loss)
      .def_readwrite("gradients", &Cross_Entropy::gradients)
      .def_readwrite("predicted", &Cross_Entropy::predicted)
      .def_readwrite("actual", &Cross_Entropy::actual);
  py::class_<Dense_Layer>(m, "Dense_Layer")
      .def(py::init<int, int, int, int>(), "Dense Layer", py::arg("height"),
           py::arg("width"), py::arg("depth"), py::arg("num_outputs"))
      .def("feed_forward", &Dense_Layer::feed_forward)
      .def("back_propagation", &Dense_Layer::back_propagation)
      .def("update_weights", &Dense_Layer::update_weights)
      .def("save_weights", &Dense_Layer::save_weights)
      .def("load_weights", &Dense_Layer::load_weights)
      .def_readwrite("gradients", &Dense_Layer::gradients)
      .def_readwrite("output", &Dense_Layer::output);
  py::class_<Max_Pool>(m, "Max_Pool")
      .def(py::init<int, int, int, int, int>(), "Max Pool Layer",
           py::arg("height"), py::arg("width"), py::arg("depth"),
           py::arg("filter_size"), py::arg("stride"))
      .def("feed_forward", &Max_Pool::feed_forward)
      .def("back_propagation", &Max_Pool::back_propagation)
      .def_readwrite("gradients", &Max_Pool::gradients)
      .def_readwrite("output", &Max_Pool::output);
  py::class_<ReLU>(m, "ReLU")
      .def(py::init<int, int, int>(), "ReLU Layer", py::arg("height"),
           py::arg("width"), py::arg("depth"))
      .def("feed_forward", &ReLU::feed_forward)
      .def("back_propagation", &ReLU::back_propagation)
      .def_readwrite("gradients", &ReLU::gradients)
      .def_readwrite("output", &ReLU::output);
  py::class_<Softmax>(m, "Softmax")
      .def(py::init<int>(), "Softmax Layer")
      .def("feed_forward", &Softmax::feed_forward)
      .def("back_propagation", &Softmax::back_propagation)
      .def_readwrite("gradients", &Softmax::gradients)
      .def_readwrite("output", &Softmax::output);
}