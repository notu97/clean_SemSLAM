#include "semantic_slam/CeresBetweenFactor.h"
#include "semantic_slam/SE3Node.h"
#include "semantic_slam/ceres_cost_terms/ceres_between.h"

#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/slam/BetweenFactor.h>

CeresBetweenFactor::CeresBetweenFactor(SE3NodePtr node0,
                                       SE3NodePtr node1,
                                       Pose3 between,
                                       Eigen::MatrixXd covariance,
                                       int tag)
  : CeresFactor(FactorType::ODOMETRY, tag)
  , between_(between)
  , covariance_(covariance)
{
    cf_ = BetweenCostTerm::Create(between, covariance);

    nodes_.push_back(node0);
    nodes_.push_back(node1);

    createGtsamFactor();
}

CeresBetweenFactor::~CeresBetweenFactor()
{
    delete cf_;
}

CeresFactor::Ptr
CeresBetweenFactor::clone() const
{
    return util::allocate_aligned<CeresBetweenFactor>(
      nullptr, nullptr, between_, covariance_, tag_);
}

void
CeresBetweenFactor::addToProblem(boost::shared_ptr<ceres::Problem> problem)
{   
    // std::cout << "CeresBetweenFactor cost function block size: " << cf_->parameter_block_sizes().size() << std::endl;
    // std::cout << "num_parameter_blocks: " << node1()->pose().data() << std::endl;
    // if (cf_->parameter_block_sizes().size() == 19) std::cout << "CeresBetweenFactor cost function block size equals19" << std::endl;
    if (cf_->parameter_block_sizes().size() == 10) std::cout << "CeresBetweenFactor cost function block size equals10" << std::endl;
    ceres::ResidualBlockId residual_id = problem->AddResidualBlock(
      cf_, NULL, node0()->pose().data(), node1()->pose().data());
    residual_ids_[problem.get()] = residual_id;

    active_ = true;
}

void
CeresBetweenFactor::createGtsamFactor() const
{
    if (!node0() || !node1())
        return;

    auto gtsam_noise = gtsam::noiseModel::Gaussian::Covariance(covariance_);

    gtsam_factor_ = util::allocate_aligned<gtsam::BetweenFactor<gtsam::Pose3>>(
      nodes_[0]->key(), nodes_[1]->key(), gtsam::Pose3(between_), gtsam_noise);
}

void
CeresBetweenFactor::addToGtsamGraph(
  boost::shared_ptr<gtsam::NonlinearFactorGraph> graph) const
{
    if (!gtsam_factor_)
        createGtsamFactor();

    graph->push_back(gtsam_factor_);
}