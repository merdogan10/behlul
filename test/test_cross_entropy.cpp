#include "cross_entropy.hpp"
#include <catch2/catch.hpp>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <vector>

using Eigen::MatrixXd;
using namespace std;

TEST_CASE("Cross Entropy Forward pass is tested", "[cross_entropy_forward]") {
  Cross_Entropy entropy(0);
  VectorXd pred(3), m1(3), m2(3);
  pred << 0.25, 0.25, 0.5;
  m1 << 1, 0, 0;
  m2 << 0, 0, 1;

  entropy.feed_forward(pred, m1);
  double loss1 = entropy.loss;
  entropy.feed_forward(pred, m2);
  double loss2 = entropy.loss;
  REQUIRE(loss1 > loss2);
}

TEST_CASE("Cross Entropy Backward pass is tested", "[cross_entropy_backward]") {
  Cross_Entropy entropy(0);
  VectorXd pred(3), m1(3), g(3);
  pred << 0.1, 0.1, 0.8;
  m1 << 0, 0, 1;
  g << 0, 0, -1.25;

  entropy.feed_forward(pred, m1);
  entropy.back_propagation();
  double epsilon = 0.00001;
  REQUIRE((entropy.gradients - g).cwiseAbs().sum() < epsilon);
}