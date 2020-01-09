/**
 * \class Cross_Entropy
 *
 *
 * \brief Cross Entropy Layer
 *
 * This class is used as Cross Entropy Layer Layer of
 * a neural network.
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
#ifndef _CROSS_ENTROPY_HPP_
#define _CROSS_ENTROPY_HPP_
#include <algorithm>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <vector>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

class Cross_Entropy {
public:
  /**
   * \brief Create a Cross_Entropy
   * \param value
   *
   * This method creates a Cross Entropy Layer.
   */
  Cross_Entropy(int value){};
  double loss;

  VectorXd predicted; ///< predicted labels
  VectorXd actual;    ///< actual labels
  VectorXd gradients; ///< gradients of the layer

  /**
   * \brief Forward pass of the Cross Entropy Layer
   * \param predicted predicted labels
   * \param actual actual labels
   *
   * This function iterates forward pass of the Cross Entropy
   * Layer. Compares predicted and actual labels and computes
   * the loss with cross entropy.
   */
  void feed_forward(VectorXd predicted, VectorXd actual);

  /**
   * \brief Backward pass of the Cross Entropy Layer
   *
   * This function iterates backward pass of the Cross Entropy
   * Layer. Computes gradients with derivative of cross entropy.
   */
  void back_propagation();
};

#endif // _CROSS_ENTROPY_HPP_