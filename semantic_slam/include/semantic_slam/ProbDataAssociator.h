#pragma once

#include "semantic_slam/DataAssociator.h"

class ProbDataAssociator : public DataAssociator
{
  public:
    ProbDataAssociator() {}
    ProbDataAssociator(ObjectParams params)
      : DataAssociator(params)
    {}

    Eigen::MatrixXd computeConstraintWeights(
      const Eigen::MatrixXd& mahals);

    Eigen::MatrixXd computeWeightsWithPermanent(
      const Eigen::MatrixXd& mahals, const Eigen::MatrixXd& covDets, int dim, bool& valid);
};
