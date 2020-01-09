/**
 * \class Softmax
 *
 *
 * \brief Softmax Layer
 *
 * This class is used as Softmax Layer of a neural network.
 *
 *
 * \author $Author: Mustafa Erdogan $
 *
 * \version $Revision: 0.87 $
 *
 * \date $Date: 24/07/2019 14:16:20 $
 *
 * Contact: mustafa.erdogan@tum.de
 *
 *
 */
#ifndef _SOFTMAX_HPP_
#define _SOFTMAX_HPP_
#include <algorithm>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <vector>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

class Softmax {
public:
  /**
   * \brief Create a Softmax
   * \param value
   *
   * This method creates a Softmax Layer.
   */
  Softmax(int value){};

  VectorXd input;     ///< input of the layer
  VectorXd output;    ///< output of the layer
  VectorXd gradients; ///< gradients of the layer

  /**
   * \brief Set input
   * \param input input of the Softmax Layer
   *
   * This method sets input of the Softmax Layer
   */
  void set_input(VectorXd input);

  /**
   * \brief Forward pass of the Softmax Layer
   * \param input input of the Softmax Layer
   *
   * This function iterates forward pass of the Softmax
   * Layer. As input it gets the previous layer's output.The
   * output filled is used later from the next layer. Computes
   * probabilities of the labels using softmax function.
   */
  void feed_forward(VectorXd input);

  /**
   * \brief Backward pass of the Softmax Layer
   * \param upstream_gradient gradients coming from the next layer
   *
   * This function iterates backward pass of the Softmax
   * Layer. As input it gets the next layer's gradients.
   */
  void back_propagation(VectorXd upstream_gradient);
};

#endif // _SOFTMAX_HPP_