/**
 * \class ReLU
 *
 *
 * \brief ReLU Layer
 *
 * This class is used as ReLU(Rectified Linear Unit) Layer
 * of a neural network. It gets details of its input initially.
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
#ifndef _RELU_HPP_
#define _RELU_HPP_
#include <algorithm>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <vector>

using namespace std;
using Eigen::MatrixXd;

class ReLU {
public:
  /**
   * \brief Create a ReLU
   * \param height height of the input
   * \param width width of the input
   * \param depth depth of the input
   *
   * This method creates a ReLU Layer. Initializes given values.
   */
  ReLU(int height, int width, int depth);
  int height; ///< height of the input
  int width;  ///< width of the input
  int depth;  ///< depth of the input

  vector<MatrixXd> input;     ///< input of the layer
  vector<MatrixXd> output;    ///< output of the layer
  vector<MatrixXd> gradients; ///< gradients of the layer

  /**
   * \brief Set input
   * \param input input of the ReLU Layer
   *
   * This method sets input of the ReLU Layer
   */
  void set_input(vector<MatrixXd> input);

  /**
   * \brief Clear output
   *
   * This method clears output of the ReLU Layer and
   * fills with zeros.
   */
  void clear_output();

  /**
   * \brief Forward pass of the ReLU Layer
   * \param input input of the ReLU Layer
   *
   * This function iterates forward pass of the ReLU
   * Layer. As input it gets the previous layer's output.The
   * output filled is used later from the next layer.
   */
  void feed_forward(vector<MatrixXd> input);

  /**
   * \brief Backward pass of the ReLU Layer
   * \param upstream_gradient gradients coming from the next layer
   *
   * This function iterates backward pass of the ReLU
   * Layer. As input it gets the next layer's gradients.
   */
  void back_propagation(vector<MatrixXd> upstream_gradient);
};

#endif // _RELU_HPP_