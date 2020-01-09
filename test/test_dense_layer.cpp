#include "dense_layer.hpp"
#include <catch2/catch.hpp>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <vector>

using Eigen::MatrixXd;
using namespace std;

TEST_CASE("Dense Layer Forward pass is tested", "[dense_layer_forward]") {
  MatrixXd i1(2, 2), i2(2, 2);
  i1 << 4, 6, 8, 2;
  i2 << 9, 7, 5, 3;

  vector<MatrixXd> input;
  input.push_back(i1);
  input.push_back(i2);
  VectorXd o(4);
  o << 4.18852, 9.53919, 2.24509, 6.19417;

  Dense_Layer dense(2, 2, 2, 4);
  MatrixXd w(4, 8);
  w << 0.804416, -0.249586, 0.0632129, 0.86162, 0.279958, -0.119791, -0.542064,
      0.912937, 0.70184, 0.520497, -0.921439, 0.441905, -0.291903, 0.76015,
      0.786745, 0.17728, -0.466669, 0.0250707, -0.124725, -0.431413, 0.375723,
      0.658402, -0.29928, 0.314608, 0.0795207, 0.335448, 0.86367, 0.477069,
      -0.668052, -0.339326, 0.37334, 0.717353;
  dense.weights = w;
  dense.feed_forward(input);
  double epsilon = 0.0001;
  REQUIRE((dense.output - o).cwiseAbs().sum() < epsilon);
}
TEST_CASE("Dense Layer Backward pass is tested", "[dense_layer_backward]") {
  MatrixXd i1(2, 2), i2(2, 2);
  i1 << 4, 6, 8, 2;
  i2 << 9, 7, 5, 3;

  vector<MatrixXd> input;
  input.push_back(i1);
  input.push_back(i2);
  VectorXd o(4);
  o << 4.18852, 9.53919, 2.24509, 6.19417;

  Dense_Layer dense(2, 2, 2, 4);
  MatrixXd w(4, 8);
  w << 0.804416, -0.249586, 0.0632129, 0.86162, 0.279958, -0.119791, -0.542064,
      0.912937, 0.70184, 0.520497, -0.921439, 0.441905, -0.291903, 0.76015,
      0.786745, 0.17728, -0.466669, 0.0250707, -0.124725, -0.431413, 0.375723,
      0.658402, -0.29928, 0.314608, 0.0795207, 0.335448, 0.86367, 0.477069,
      -0.668052, -0.339326, 0.37334, 0.717353;
  dense.weights = w;
  dense.feed_forward(input);
  VectorXd upstream_gradient = VectorXd::Ones(4);
  dense.back_propagation(upstream_gradient);
  MatrixXd g0(2, 2), g1(2, 2);
  g0 << 1.11911, -0.119281, 0.63143, 1.34918;
  g1 << -0.304274, 0.318741, 0.959435, 2.12218;
  double epsilon = 0.0001;
  REQUIRE((dense.gradients[0] - g0).cwiseAbs().sum() < epsilon);
  REQUIRE((dense.gradients[1] - g1).cwiseAbs().sum() < epsilon);
}