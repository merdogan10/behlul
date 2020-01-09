/**
 * \class Max_Pool
 *
 *
 * \brief Maximum Pooling Layer
 *
 * This class is used as Maximum Pooling Layer of a neural
 * network. It gets details of its input and filter initially.
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
#ifndef _MAX_POOL_HPP_
#define _MAX_POOL_HPP_
#include <algorithm>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <vector>

using namespace std;
using Eigen::MatrixXd;

class Max_Pool {
public:
  /**
   * \brief Create a Max_Pool
   * \param height height of the input
   * \param width width of the input
   * \param depth depth of the input
   * \param filter_size height and width of the square filter
   * \param stride stride of the filter
   *
   * This method creates a Maximum Pooling Layer. Initializes given values.
   */
  Max_Pool(int height, int width, int depth, int filter_size, int stride);
  int height;          ///< height of the input
  int width;           ///< width of the input
  int depth;           ///< depth of the input
  int filter_size = 2; ///< height and width of the square filter
  int stride = 2;      ///< stride of the filter

  vector<MatrixXd> input;     ///< input of the layer
  vector<MatrixXd> output;    ///< output of the layer
  vector<MatrixXd> gradients; ///< gradients of the layer

  /**
   * \brief Set input
   * \param input input of the Maximum Pooling Layer
   *
   * This method sets input of the Maximum Pooling Layer
   */
  void set_input(vector<MatrixXd> input);

  /**
   * \brief Clear output
   *
   * This method clears output of the Maximum Pooling Layer and
   * fills with zeros.
   */
  void clear_output();

  /**
   * \brief Forward pass of the Maximum Pooling Layer
   * \param input input of the Maximum Pooling Layer
   *
   * This function iterates forward pass of the Maximum Pooling
   * Layer. As input it gets the previous layer's output.The
   * output filled is used later from the next layer.
   */
  void feed_forward(vector<MatrixXd> input);

  /**
   * \brief Backward pass of the Maximum Pooling Layer
   * \param upstream_gradient gradients coming from the next layer
   *
   * This function iterates backward pass of the Maximum Pooling
   * Layer. As input it gets the next layer's gradients.
   */
  void back_propagation(vector<MatrixXd> upstream_gradient);
};

#endif // _MAX_POOL_HPP_