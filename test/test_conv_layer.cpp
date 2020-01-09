#include "conv_layer.hpp"
#include <catch2/catch.hpp>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <vector>

using Eigen::MatrixXd;
using namespace std;

TEST_CASE("Conv Layer Forward pass is tested", "[conv_layer_forward]") {
  MatrixXd i1(7, 7);
  MatrixXd i2(7, 7);
  MatrixXd i3(7, 7);
  i1 << 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 2, 2, 0, 0, 0, 1, 2,
      1, 0, 0, 0, 0, 1, 2, 2, 2, 0, 0, 0, 1, 0, 1, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0;

  i2 << 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 1, 0, 0, 0, 2, 1, 0, 2, 0, 0, 2, 1,
      1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0;

  i3 << 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 1, 2, 1, 0, 0, 1, 2, 2, 0, 1, 0, 0, 2, 1,
      0, 0, 0, 0, 0, 2, 0, 0, 1, 1, 0, 0, 0, 1, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0;

  vector<MatrixXd> input;
  input.push_back(i1);
  input.push_back(i2);
  input.push_back(i3);

  MatrixXd o0(3, 3);
  MatrixXd o1(3, 3);
  o0 << -3, -5, 5, -5, -1, 4, 0, 0, -2;
  o1 << 0, -2, 4, 1, 13, 3, 2, 6, 6;

  int stride = 2, num_filters = 2, filter_size = 3;
  Conv_Layer *cl = new Conv_Layer(7, 7, 3, filter_size, stride, num_filters);
  cl->filters.clear();
  cl->filters.resize(num_filters);

  MatrixXd w00(filter_size, filter_size);
  MatrixXd w01(filter_size, filter_size);
  MatrixXd w02(filter_size, filter_size);
  MatrixXd w10(filter_size, filter_size);
  MatrixXd w11(filter_size, filter_size);
  MatrixXd w12(filter_size, filter_size);
  w00 << 0, 1, 0, -1, 0, -1, 1, 0, -1;
  w01 << -1, 1, -1, 0, 0, -1, -1, 1, -1;
  w02 << -1, 1, 1, 1, 0, 0, -1, 0, 1;
  w10 << 1, 1, 1, 1, 0, -1, 0, 1, 1;
  w11 << -1, 1, 1, 1, 0, -1, -1, 1, 0;
  w12 << 1, 0, 0, 0, 0, 0, -1, -1, 1;
  cl->filters[0].push_back(w00);
  cl->filters[0].push_back(w01);
  cl->filters[0].push_back(w02);
  cl->filters[1].push_back(w10);
  cl->filters[1].push_back(w11);
  cl->filters[1].push_back(w12);

  cl->feed_forward(input);
  REQUIRE(cl->output[0].isApprox(o0));
  REQUIRE(cl->output[1].isApprox(o1));
}

TEST_CASE("Conv Layer Backward pass is tested", "[test_conv_layer_backward]") {
  MatrixXd i1(3, 3);
  i1 << 1, 2, 3, 2, 3, 4, 3, 4, 5;
  vector<MatrixXd> input;
  input.push_back(i1);
  int stride = 1, num_filters = 2, filter_size = 2;
  Conv_Layer *cl = new Conv_Layer(3, 3, 1, filter_size, stride, num_filters);
  cl->filters.clear();
  cl->filters.resize(num_filters);

  MatrixXd w0(filter_size, filter_size);
  MatrixXd w1(filter_size, filter_size);
  w0 << 1, 0, 0, 1;
  w1 << 0, 1, 1, 0;
  cl->filters[0].push_back(w0);
  cl->filters[1].push_back(w1);

  cl->feed_forward(input);

  vector<MatrixXd> upstream_gradient;
  upstream_gradient.push_back(MatrixXd::Ones(2, 2));
  upstream_gradient.push_back(MatrixXd::Ones(2, 2));
  cl->back_propagation(upstream_gradient);
  MatrixXd g(3, 3);
  g << 1, 2, 1, 2, 4, 2, 1, 2, 1;
  REQUIRE(cl->gradients[0].isApprox(g));
}