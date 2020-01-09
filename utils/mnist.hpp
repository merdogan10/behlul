/**
 * \class MNIST
 *
 *
 * \brief MNIST dataset reader
 *
 * This class is used to get dataset for MNIST.
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
#ifndef _MNIST_HPP_
#define _MNIST_HPP_
#include <algorithm>
#include <eigen3/Eigen/Dense>
#include <fstream>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <vector>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

class MNIST {
public:
  /**
   * \brief Create a MNIST
   * \param dir folder path to get MNIST dataset
   *
   * This method creates a MNIST. It provides train data,
   * validation data, test data and train labels, validation labels
   */
  MNIST(string dir);
  string dir = "data"; ///< folder path to get MNIST dataset
  string train_file;   ///< file path to get MNIST train dataset
  string test_file;    ///< file path to get MNIST test dataset
  double split_ratio =
      0.9; ///< split ratio of the train dataset to train and validation

  vector<vector<MatrixXd>> train_data;      ///< train data
  vector<vector<MatrixXd>> validation_data; ///< validation data
  vector<vector<MatrixXd>> test_data;       ///< test data

  vector<VectorXd> train_labels;      ///< train labels
  vector<VectorXd> validation_labels; ///< validation labels
};

#endif // _MNIST_HPP_