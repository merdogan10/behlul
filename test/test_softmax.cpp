#include "softmax.hpp"
#include <catch2/catch.hpp>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <vector>

using Eigen::MatrixXd;
using namespace std;

TEST_CASE("Softmax Layer Forward pass is tested", "[softmax_forward]") {
  VectorXd m(5), o(5);
  m << 10, 9, 7, 11, 8;
  o << 0.234122, 0.0861285, 0.0116562, 0.636409, 0.0316849;
  Softmax soft(0);
  soft.feed_forward(m);
  double epsilon = 0.00001;
  REQUIRE((soft.output - o).cwiseAbs().sum() < epsilon);
}

TEST_CASE("Softmax Layer Backward pass is tested", "[softmax_backward]") {
  VectorXd m(5), o(5);
  m << 10, 9, 7, 11, 8;
  o << 0.234122, 0.0861285, 0.0116562, 0.636409, 0.0316849;
  Softmax soft(0);
  soft.feed_forward(m);
  VectorXd upstream_gradient(5), g(5);
  upstream_gradient << 1, 3, 5, 7, 9;
  g << -1.00457, -0.197304, -0.00338979, 1.08774, 0.117525;
  soft.back_propagation(upstream_gradient);
  double epsilon = 0.00001;
  REQUIRE((soft.gradients - g).cwiseAbs().sum() < epsilon);
}