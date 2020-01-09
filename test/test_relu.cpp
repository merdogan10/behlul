#include "relu.hpp"
#include <catch2/catch.hpp>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <vector>

using Eigen::MatrixXd;
using namespace std;

TEST_CASE("ReLU Layer Forward pass is tested", "[relu_forward]") {
  MatrixXd m0(3, 3), m1(3, 3), m2(3, 3), o0(3, 3), o1(3, 3), o2(3, 3);
  m0 << 0.780465, -0.959954, -0.52344, -0.302214, -0.0845965, 0.941268,
      -0.871657, -0.873808, 0.804416;
  o0 << 0.780465, 0, 0, 0, 0, 0.941268, 0, 0, 0.804416;
  m1 << 0.70184, -0.249586, 0.335448, -0.466669, 0.520497, 0.0632129, 0.0795207,
      0.0250707, -0.921439;

  o1 << 0.70184, 0, 0.335448, 0, 0.520497, 0.0632129, 0.0795207, 0.0250707, 0;

  m2 << -0.124725, 0.441905, 0.279958, 0.86367, -0.431413, -0.291903, 0.86162,
      0.477069, 0.375723;

  o2 << 0, 0.441905, 0.279958, 0.86367, 0, 0, 0.86162, 0.477069, 0.375723;

  vector<MatrixXd> input;
  input.push_back(m0);
  input.push_back(m1);
  input.push_back(m2);
  ReLU *rl = new ReLU(3, 3, 3);
  rl->feed_forward(input);
  REQUIRE(rl->output[0].isApprox(o0));
  REQUIRE(rl->output[1].isApprox(o1));
  REQUIRE(rl->output[2].isApprox(o2));
}

TEST_CASE("ReLU Layer Backward pass is tested", "[relu_backward]") {
  MatrixXd m0(3, 3), m1(3, 3), m2(3, 3), o0(3, 3), o1(3, 3), o2(3, 3);
  m0 << 0.780465, -0.959954, -0.52344, -0.302214, -0.0845965, 0.941268,
      -0.871657, -0.873808, 0.804416;
  o0 << 0.780465, 0, 0, 0, 0, 0.941268, 0, 0, 0.804416;
  m1 << 0.70184, -0.249586, 0.335448, -0.466669, 0.520497, 0.0632129, 0.0795207,
      0.0250707, -0.921439;

  o1 << 0.70184, 0, 0.335448, 0, 0.520497, 0.0632129, 0.0795207, 0.0250707, 0;

  m2 << -0.124725, 0.441905, 0.279958, 0.86367, -0.431413, -0.291903, 0.86162,
      0.477069, 0.375723;

  o2 << 0, 0.441905, 0.279958, 0.86367, 0, 0, 0.86162, 0.477069, 0.375723;

  vector<MatrixXd> input;
  input.push_back(m0);
  input.push_back(m1);
  input.push_back(m2);
  ReLU *rl = new ReLU(3, 3, 3);
  rl->feed_forward(input);

  MatrixXd g0(3, 3), g1(3, 3), g2(3, 3);
  g0 << 1, 0, 0, 0, 0, 1, 0, 0, 1;
  g1 << 1, 0, 1, 0, 1, 1, 1, 1, 0;
  g2 << 0, 1, 1, 1, 0, 0, 1, 1, 1;
  vector<MatrixXd> upstream_gradient;
  upstream_gradient.push_back(MatrixXd::Ones(3, 3));
  upstream_gradient.push_back(MatrixXd::Ones(3, 3));
  upstream_gradient.push_back(MatrixXd::Ones(3, 3));
  rl->back_propagation(upstream_gradient);
  REQUIRE(rl->gradients[0].isApprox(g0));
  REQUIRE(rl->gradients[1].isApprox(g1));
  REQUIRE(rl->gradients[2].isApprox(g2));
}