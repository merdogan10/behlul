#include "max_pool.hpp"
#include <catch2/catch.hpp>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <vector>

using Eigen::MatrixXd;
using namespace std;

TEST_CASE("Max Pool Layer Forward pass is tested", "[max_pool_forward]") {
  MatrixXd m(4, 4), o(2, 2);
  m << 1, 1, 2, 4, 5, 6, 7, 8, 3, 2, 1, 0, 1, 2, 3, 4;
  o << 6, 8, 3, 4;
  vector<MatrixXd> input;
  input.push_back(m);
  Max_Pool *mp = new Max_Pool(4, 4, 1, 2, 2);
  mp->feed_forward(input);
  REQUIRE(mp->output[0].isApprox(o));
}

TEST_CASE("Max Pool Layer Backward pass is tested", "[max_pool_backward]") {
  MatrixXd m(4, 4), o(2, 2);
  m << 1, 1, 2, 4, 5, 6, 7, 8, 3, 2, 1, 0, 1, 2, 3, 4;
  o << 6, 8, 3, 4;
  vector<MatrixXd> input;
  input.push_back(m);
  Max_Pool *mp = new Max_Pool(4, 4, 1, 2, 2);
  mp->feed_forward(input);
  vector<MatrixXd> upstream_gradient;
  upstream_gradient.push_back(MatrixXd::Ones(4, 4));
  mp->back_propagation(upstream_gradient);
  MatrixXd g(4, 4);
  g << 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1;
  REQUIRE(mp->gradients[0].isApprox(g));
}
