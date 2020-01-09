/**
 * \class Dense_Layer
 *
 *
 * \brief Dense Layer
 *
 * This class is used as Dense Layer of a neural network.
 * It gets details of its input and output.
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
#ifndef _DENSE_LAYER_HPP_
#define _DENSE_LAYER_HPP_
#include <algorithm>
#include <eigen3/Eigen/Dense>
#include <fstream>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <vector>

using namespace std;
using Eigen::Map;
using Eigen::MatrixXd;
using Eigen::RowVectorXd;
using Eigen::VectorXd;

class Dense_Layer {
public:
  /**
   * \brief Create a Dense_Layer
   * \param height height of the input
   * \param width width of the input
   * \param depth depth of the input
   * \param num_outputs number of output classes
   *
   * This method creates a Dense Layer. Initializes given values.
   * Initializes weights with uniform random values.
   */
  Dense_Layer(int height, int width, int depth, int num_outputs);
  int height;      ///< height of the input
  int width;       ///< width of the input
  int depth;       ///< depth of the input
  int num_outputs; ///< number of output classes

  VectorXd output;        ///< output of the layer
  vector<MatrixXd> input; ///< input of the layer
  MatrixXd weights;       ///< weights of the layer
  VectorXd biases;        ///< biases of the layer

  vector<MatrixXd> gradients; ///< gradients of the layer
  MatrixXd gradient_weights;  ///< gradients of the weights
  VectorXd gradient_biases;   ///< gradients of the biases

  vector<MatrixXd>
      accumulated_gradients; ///< accumulated gradients of the layer
  MatrixXd
      accumulated_gradient_weights; ///< accumulated gradients of the weights
  VectorXd accumulated_gradient_biases; ///< accumulated gradients of the biases

  /**
   * \brief Set input
   * \param input input of the Dense Layer
   *
   * This method sets input of the Dense Layer
   */
  void set_input(vector<MatrixXd> input);

  /**
   * \brief Clear output
   *
   * This method clears output of the Dense Layer and
   * fills with zeros.
   */
  void clear_output();

  /**
   * \brief Forward pass of the Dense Layer
   * \param input input of the Dense Layer
   *
   * This function iterates forward pass of the Dense
   * Layer. As input it gets the previous layer's output and convert
   * to a dense input, then uses the input.The output filled is used
   * later from the next layer.
   */
  void feed_forward(vector<MatrixXd> input);

  /**
   * \brief Backward pass of the Dense Layer
   * \param upstream_gradient gradients coming from the next layer
   *
   * This function iterates backward pass of the Dense
   * Layer. As input it gets the next layer's gradients.
   */
  void back_propagation(VectorXd upstream_gradient);

  /**
   * \brief Update weights of the Dense Layer
   * \param batch_size batch size
   * \param learning_rate learning rate
   *
   * This function updates weights of the Dense Layer.
   */
  void update_weights(int batch_size, double learning_rate);

  /**
   * \brief Save weights to local file
   * \param dir file path to save weights
   *
   * This function saves weights of the Convolutional
   * Layer to use later in pretrained demo.
   */
  void save_weights(string dir);

  /**
   * \brief Load weights from local file
   * \param dir file path to load weights
   *
   * This function loads weights of the Convolutional
   * Layer to use in pretrained demo.
   */
  void load_weights(string dir);
};

#endif // _DENSE_LAYER_HPP_