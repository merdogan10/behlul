/**
 * \class Conv_Layer
 *
 *
 * \brief Convolutional Layer
 *
 * This class is used as Convolutional Layer of a neural network.
 * It gets details of its input and filter initially.
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
#ifndef _CONV_LAYER_HPP_
#define _CONV_LAYER_HPP_
#include <algorithm>
#include <eigen3/Eigen/Dense>
#include <fstream>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <vector>

using namespace std;
using Eigen::MatrixXd;

class Conv_Layer {
public:
  /**
   * \brief Create a Conv_Layer
   * \param height height of the input
   * \param width width of the input
   * \param depth depth of the input
   * \param filter_size height and width of the square filter
   * \param stride stride of the filter
   * \param num_filters number of filters
   *
   * This method creates a Convolutional Layer. Initializes given values.
   * Initializes filters with uniform random values.
   */
  Conv_Layer(int height, int width, int depth, int filter_size, int stride,
             int num_filters);
  int height;          ///< height of the input
  int width;           ///< width of the input
  int depth;           ///< depth of the input
  int filter_size = 3; ///< height and width of the square filter
  int stride = 1;      ///< stride of the filter
  int num_filters = 1; ///< number of filters

  vector<MatrixXd> input;                    ///< input of the layer
  vector<MatrixXd> output;                   ///< output of the layer
  vector<MatrixXd> gradients;                ///< gradients of the layer
  vector<vector<MatrixXd>> filters;          ///< filters of the layer
  vector<vector<MatrixXd>> gradient_filters; ///< gradients of the filters
  vector<vector<MatrixXd>>
      accumulated_gradient_filters; ///< accumulated gradients of the filters

  /**
   * \brief Set input
   * \param input input of the Convolutional Layer
   *
   * This method sets input of the Convolutional Layer
   */
  void set_input(vector<MatrixXd> input);

  /**
   * \brief Clear output
   *
   * This method clears output of the Convolutional Layer and
   * fills with zeros.
   */
  void clear_output();

  /**
   * \brief Forward pass of the Convolutional Layer
   * \param input input of the Convolutional Layer
   *
   * This function iterates forward pass of the Convolutional
   * Layer. As input it gets the previous layer's output in middle layers
   * or image in starting layer. The output filled is used later from
   * the next layer.
   */
  void feed_forward(vector<MatrixXd> input);

  /**
   * \brief Backward pass of the Convolutional Layer
   * \param upstream_gradient gradients coming from the next layer
   *
   * This function iterates backward pass of the Convolutional
   * Layer. As input it gets the next layer's gradients.
   */
  void back_propagation(vector<MatrixXd> upstream_gradient);

  /**
   * \brief Update filters of the Convolutional Layer
   * \param batch_size batch size
   * \param learning_rate learning rate
   *
   * This function updates filters of the Convolutional
   * Layer.
   */
  void update_weights(int batch_size, double learning_rate);

  /**
   * \brief Save filters to local file
   * \param dir file path to save filters
   *
   * This function saves filters of the Convolutional
   * Layer to use later in pretrained demo.
   */
  void save_filters(string dir);

  /**
   * \brief Load filters from local file
   * \param dir file path to load filters
   *
   * This function loads filters of the Convolutional
   * Layer to use in pretrained demo.
   */
  void load_filters(string dir);
};

#endif // _CONV_LAYER_HPP_