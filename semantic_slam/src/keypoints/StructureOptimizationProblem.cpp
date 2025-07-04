#include "semantic_slam/keypoints/StructureOptimizationProblem.h"

#include "semantic_slam/Symbol.h"
#include <algorithm>

#include "semantic_slam/Pose3.h"
#include "semantic_slam/SE3Node.h"
#include "semantic_slam/SemanticKeyframe.h"
#include "semantic_slam/ceres_cost_terms/ceres_pose_prior.h"
#include "semantic_slam/ceres_cost_terms/ceres_projection.h"
#include "semantic_slam/ceres_cost_terms/ceres_bbox_projection.h"
#include "semantic_slam/ceres_cost_terms/ceres_structure.h"
// #include "semantic_slam/keypoints/ceres_camera_constraint.h"
#include "semantic_slam/keypoints/geometry.h"

#include <rosfmt/rosfmt.h>

void
StructureOptimizationProblem::setBasisCoefficients(
  const Eigen::VectorXd& coeffs)
{
    basis_coefficients_ = coeffs;
}

Eigen::VectorXd
StructureOptimizationProblem::getBasisCoefficients() const
{
    return basis_coefficients_;
}

boost::shared_ptr<Eigen::Vector3d>
StructureOptimizationProblem::getKeypoint(size_t index) const
{
    return kps_[index];
}

Pose3
StructureOptimizationProblem::getObjectPose() const
{
    return object_pose_;
}

Eigen::MatrixXd
StructureOptimizationProblem::getPlx(size_t camera_index)
{
    size_t Plx_dim = 6 + 3 * m_;

    Eigen::MatrixXd cov = Eigen::MatrixXd::Zero(Plx_dim, Plx_dim);

    double buf[36];

    if (!have_covariance_) {
        computeCovariances();
    }

    // if (have_covariance_) {
    //   // Just get Pll for now...
    //   covariance_->GetCovarianceBlock(kps_[landmark_index]->data(),
    //                                   kps_[landmark_index]->data(), Pll);
    //   cov.block<3, 3>(0, 0) =
    //     Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(Pll);

    //   return cov;
    // }

    if (!have_covariance_) {
        cov = Eigen::MatrixXd::Identity(Plx_dim, Plx_dim);
        cov.bottomRightCorner<3, 3>() = Eigen::Matrix3d::Zero();
        return cov;
    }

    // what's crazy is that sometimes camera_index won't be in our list of
    // optimized camera poses.
    // for many purposes we just need one *near* it so use the closest one
    // and warn about it
    if (keyframes_.find(camera_index) == keyframes_.end()) {
        size_t closest_index = 0;
        size_t index_distance = std::numeric_limits<size_t>::max();

        for (auto& frame_pair : keyframes_) {
            // std::cout << "Index " << frame_pair.first << " has distance " <<
            // std::abs((int)camera_index - (int)frame_pair.first) << std::endl;
            if (std::abs((int)camera_index - (int)frame_pair.first) <
                index_distance) {
                closest_index = frame_pair.first;
                index_distance =
                  std::abs((int)closest_index - (int)frame_pair.first);
            }
        }

        // ROS_WARN_STREAM("WARNING: Camera pose " << camera_index << " not in
        // optimization values. Using "
        //                   << closest_index << " instead.");

        camera_index = closest_index;
    }

    std::vector<double*> parameter_blocks;
    std::vector<size_t> block_sizes;
    for (auto& kp : kps_) {
        parameter_blocks.push_back(kp->data());
        block_sizes.push_back(3);
    }

    parameter_blocks.push_back(local_pose_nodes_[camera_index]->pose().data());
    block_sizes.push_back(6); // <-- 6 because blocks are in the tangent space

    size_t index_i = 0;
    size_t index_j = 0;

    using RowMajorMatrixXd =
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    for (size_t i = 0; i < parameter_blocks.size(); ++i) {
        for (size_t j = i; j < parameter_blocks.size(); ++j) {
            covariance_->GetCovarianceBlockInTangentSpace(
              parameter_blocks[i], parameter_blocks[j], buf);
            cov.block(index_i, index_j, block_sizes[i], block_sizes[j]) =
              Eigen::Map<RowMajorMatrixXd>(buf, block_sizes[i], block_sizes[j]);

            index_j += block_sizes[j];
        }

        index_i += block_sizes[i];
        index_j = index_i;
    }

    return cov.selfadjointView<Eigen::Upper>();
}

StructureOptimizationProblem::StructureOptimizationProblem(
  geometry::ObjectModelBasis model,
  boost::shared_ptr<CameraCalibration> camera_calibration,
  Pose3 body_T_camera,
  Eigen::VectorXd weights,
  ObjectParams params)
  : model_(model)
  , camera_calibration_(camera_calibration)
  , body_T_camera_(body_T_camera)
  , object_pose_(Pose3::Identity())
  , weights_(weights)
  // , num_poses_(0)
  , params_(params)
  , solved_(false)
  , have_covariance_(false)
  , random_generator_(random_device_())
{
    m_ = model_.mu.cols();
    k_ = model_.pc.rows() / 3;
    
    // object_shape_ = model_.mu.rowwise().maxCoeff() - model_.mu.rowwise().minCoeff();
    object_shape_ = Eigen::Vector3d{1.548, 1.623, 3.893};
    // std::cout << "object_shape: [" << object_shape_.rows() << ", " << object_shape_.cols() << "]\n";
    // std::cout << "object_shape: "  << object_shape_(0) << " " 
    //                                << object_shape_(1) << " " 
    //                                << object_shape_(2) << std::endl;

    ceres_problem_ = boost::make_shared<ceres::Problem>();

    basis_coefficients_ = Eigen::VectorXd::Zero(k_);

    // ceres::Problem will take ownership of this pointer
    pose_parameterization_ = new SE3Node::Parameterization;

    for (size_t i = 0; i < m_; ++i) {
        kps_.push_back(util::allocate_aligned<Eigen::Vector3d>());
    }

    // weights_ = Eigen::VectorXd::Ones(m_);

    structure_cf_ = StructureCostTerm::Create(
      model_, weights_, params_.structure_regularization_factor);

    // Accumulate data pointers for ceres parameters
    ceres_parameters_.push_back(object_pose_.data());
    for (size_t i = 0; i < m_; ++i) {
        ceres_parameters_.push_back(kps_[i]->data());
    }
    if (k_ > 0) {
        ceres_parameters_.push_back(basis_coefficients_.data());
    }

    ceres::LossFunction* structure_loss = new ceres::ScaledLoss(
      NULL, params_.structure_error_coefficient, ceres::TAKE_OWNERSHIP);
    // std::cout << "StructureOptimization cost function 1 block size: " << structure_cf_->parameter_block_sizes().size() << std::endl;
    // if (structure_cf_->parameter_block_sizes().size() == 10) std::cout << "StructureOptimization cost function 1 block size equals10 " << std::endl;
    ceres_problem_->AddResidualBlock(
      structure_cf_, structure_loss, ceres_parameters_);
    ceres_problem_->SetParameterization(object_pose_.data(),
                                       pose_parameterization_);
}

void
StructureOptimizationProblem::setRotation(const Eigen::Quaterniond& G_q_O)
{
    Pose3 new_pose = object_pose_;
    new_pose.rotation() = G_q_O;

    for (auto& kp : kps_) {
        *kp = new_pose.transform_from(object_pose_.transform_to(*kp));
    }

    object_pose_ = new_pose;
}

Eigen::Quaterniond
StructureOptimizationProblem::randomUniformQuaternion()
{
    std::uniform_real_distribution<> dist(0.0, 1.0);

    double s = dist(random_generator_);
    double sigma1 = std::sqrt(1 - s);
    double sigma2 = std::sqrt(s);

    double theta1 = 3.14159 * 2 * dist(random_generator_);
    double theta2 = 3.14159 * 2 * dist(random_generator_);

    Eigen::Quaterniond q(std::cos(theta2) * sigma2,
                         std::sin(theta1) * sigma1,
                         std::cos(theta1) * sigma1,
                         std::sin(theta2) * sigma2);

    if (q.w() < 0)
        q.coeffs() *= -1;

    return q.normalized();
}

void
StructureOptimizationProblem::addKeypointMeasurement(
  const KeypointMeasurement& kp_msmt)
{
    // size_t pose_index;
    // if (camera_pose_ids_.find(kp_msmt.measured_symbol.index()) ==
    //     camera_pose_ids_.end()) {
    //   camera_pose_ids_[kp_msmt.measured_symbol.index()] = num_poses_;
    //   num_poses_++;
    // }

    // ceres::LossFunction* cauchy_loss =
    //   new ceres::CauchyLoss(params_.robust_estimator_parameter);

    int index = Symbol(kp_msmt.measured_key).index();

    ceres::LossFunction* huber_loss =
      new ceres::HuberLoss(params_.robust_estimator_parameter);

    // double effective_sigma = kp_msmt.pixel_sigma / kp_msmt.score;

    double effective_sigma = kp_msmt.pixel_sigma;

    Eigen::Matrix2d msmt_covariance =
      effective_sigma * effective_sigma * Eigen::Matrix2d::Identity();

    ceres::CostFunction* projection_cf =
      ProjectionCostTerm::Create(kp_msmt.pixel_measurement,
                                 msmt_covariance,
                                 body_T_camera_.rotation(),
                                 body_T_camera_.translation(),
                                 camera_calibration_);

    // ceres_problem_->AddResidualBlock(projection_cf, NULL,
    // object_pose_.rotation_data(), object_pose_.translation_data(),
    // kps_[kp_msmt.kp_class_id]->data());
    auto& pose_node = local_pose_nodes_[index];
    // std::cout << "StructureOptimization cost function 2 block size: " << projection_cf->parameter_block_sizes().size() << std::endl;
    // std::cout << "Param num: " << 7 + kps_[kp_msmt.kp_class_id].size() << std::endl;
    // if (projection_cf->parameter_block_sizes().size() == 19) std::cout << "StructureOptimization cost function 2 equals19" << std::endl;
    // if (projection_cf->parameter_block_sizes().size() == 10) std::cout << "StructureOptimization cost function 2 equals10" << std::endl;
    projection_residual_ids_[index] =
      ceres_problem_->AddResidualBlock(projection_cf,
                                      huber_loss,
                                      pose_node->pose().data(),
                                      kps_[kp_msmt.kp_class_id]->data());

    // // Add depth if available
    // if (params_.include_depth_constraints && kp_msmt.measured_depth > 0) {
    //   double depth_covariance =
    //     kp_msmt.measured_depth_sigma * kp_msmt.measured_depth_sigma;
    //   ceres::CostFunction* range_cf = RangeCostTerm::Create(
    //     kp_msmt.measured_depth, depth_covariance, body_T_camera_.rotation(),
    //     body_T_camera_.translation());

    //   ceres_problem_->AddResidualBlock(
    //     range_cf, cauchy_loss, cam_pose.rotation_data(),
    //     cam_pose.translation_data(), kps_[kp_msmt.kp_class_id]->data());
    // }

    solved_ = false;
    have_covariance_ = false;
}

void
StructureOptimizationProblem::addBboxMeasurement(
  const Eigen::Vector3d& bbox_line, size_t index)
{ 
  Eigen::Matrix3d K;
  double fx = camera_calibration_->fx(), cx = camera_calibration_->u0(),
         fy = camera_calibration_->fy(), cy = camera_calibration_->v0();
  K << fx, 0, cx,
       0, fy, cy,
       0, 0, 1;

  ceres::LossFunction* huber_loss =
      new ceres::HuberLoss(params_.robust_estimator_parameter);

  std::cout << "creating bbox projection cost function" << std::endl;
  ceres::CostFunction* bbox_projection_cf = 
    ProjectionCostTermBbox::Create(bbox_line, 
                                   object_shape_, 
                                   body_T_camera_.rotation(),
                                   body_T_camera_.translation(),
                                   K);

  auto& pose_node = local_pose_nodes_[index];
  ceres_problem_->AddResidualBlock(bbox_projection_cf,
                                   huber_loss,
                                   pose_node->pose().data(),
                                   object_pose_.data());

}

void
StructureOptimizationProblem::removeKeypointMeasurement(
  const KeypointMeasurement& kp_msmt)
{
    int index = Symbol(kp_msmt.measured_key).index();
    auto residual_it = projection_residual_ids_.find(index);

    if (residual_it != projection_residual_ids_.end()) {
        ceres_problem_->RemoveResidualBlock(residual_it->second);
        projection_residual_ids_.erase(residual_it);
    }
}

double
StructureOptimizationProblem::solveWithRestarts()
{
    // We have been initialized with the result of optimization from a single
    // frame, probably. This often results in landing in a local minimum for the
    // resulting orientation... try a few different starting orientations.
    std::vector<double> costs;
    aligned_vector<Eigen::Quaterniond> orientations;

    int n_restarts = 8;

    // including, of course, the initial orientation already given
    orientations.push_back(object_pose_.rotation());
    double cost0 = solve();
    costs.push_back(cost0);

    for (int i = 0; i < n_restarts; ++i) {
        setRotation(Eigen::Quaterniond::UnitRandom());

        orientations.push_back(object_pose_.rotation());
        costs.push_back(solve());
    }

    int best_index = 0;
    double best_cost = std::numeric_limits<double>::max();
    for (size_t i = 0; i < costs.size(); ++i) {
        if (costs[i] < best_cost) {
            best_cost = costs[i];
            best_index = i;
        }
    }

    setRotation(orientations[best_index]);
    return solve();
}

double
StructureOptimizationProblem::solve()
{
    ceres::Solver::Options options;
    // options.linear_solver_type = ceres::DENSE_QR;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    // options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summary;
    ceres::Solve(options, ceres_problem_.get(), &summary);

    // std::cout << summary.FullReport() << "\n";
    // std::cout << summary.BriefReport() << "\n";
    // for (size_t i = 0; i < kp_ptrs.size(); ++i)
    // {
    //   std::cout << "kp " << i << ": " << initial_kps[i].transpose() << " -> "
    //   << kp_ptrs[i]->transpose() << std::endl;
    // }

    // Eigen::Matrix3d R = math::quat2rot(object_pose_.rotation());

    // std::cout << "R = \n" << R << std::endl;
    // std::cout << "t = \n" << object_pose_.translation() << std::endl;

    solved_ = true;

    return summary.final_cost;
}

// void
// StructureOptimizationProblem::addAllBlockPairs(
//   const std::vector<const double*>& to_add, CovarianceBlocks& blocks) const
// {
//   for (size_t i = 0; i < to_add.size(); ++i) {
//     for (size_t j = i + 1; j < to_add.size(); ++j) {
//       blocks.push_back(std::make_pair(to_add[i], to_add[j]));
//     }
//   }
// }

void
StructureOptimizationProblem::computeCovariances()
{
    ceres::Covariance::Options cov_options;

    // NOTE: with this false, the structure loss is artificially small even
    // though we may not want the cauchy losses on the projections applied here.
    // TODO modify structure cost fn so this isn't the case
    cov_options.apply_loss_function = true;

    // cov_options.num_threads = 2; // can tweak this

    cov_options.sparse_linear_algebra_library_type =
      ceres::SUITE_SPARSE; // Eigen is SLOW

    // ceres::Covariance covariance(cov_options);

    covariance_ = boost::make_shared<ceres::Covariance>(cov_options);

    std::vector<const double*> blocks;
    for (auto& node_pair : local_pose_nodes_) {
        blocks.push_back(node_pair.second->pose().data());
    }

    for (auto& kp : kps_) {
        blocks.push_back(kp->data());
    }

    blocks.push_back(object_pose_.data());

    bool succeeded = covariance_->Compute(blocks, ceres_problem_.get());

    if (!succeeded) {
        ROS_WARN_STREAM("Covariance computation failed");
        ROS_WARN_STREAM("Problem has "
                        << ceres_problem_->NumParameters() << " parameters and "
                        << ceres_problem_->NumResiduals() << " residuals.");
        have_covariance_ = false;
    } else {
        have_covariance_ = true;
    }
}

void
StructureOptimizationProblem::addCamera(SemanticKeyframe::Ptr keyframe,
                                        bool use_constant_camera_pose)
{
    if (keyframes_.find(keyframe->index()) == keyframes_.end()) {
        keyframes_.emplace(keyframe->index(), keyframe);
        local_pose_nodes_.emplace(
          keyframe->index(), util::allocate_aligned<SE3Node>(keyframe->key()));

        auto& node = local_pose_nodes_[keyframe->index()];
        node->pose() = keyframe->pose();

        node->addToProblem(ceres_problem_);

        if (!use_constant_camera_pose) {
            ceres::CostFunction* pose_prior_cf = PosePriorCostTerm::Create(
              keyframe->pose(), keyframe->covariance());
            // std::cout << "StructureOptimization cost function 3 block size: " << pose_prior_cf->parameter_block_sizes().size() << std::endl;
            // if (pose_prior_cf->parameter_block_sizes().size() == 19) std::cout << "StructureOptimization cost function 3 equals19" << std::endl;
            // if (pose_prior_cf->parameter_block_sizes().size() == 10) std::cout << "StructureOptimization cost function 3 equals10" << std::endl;
            prior_residual_ids_[keyframe->index()] =
              ceres_problem_->AddResidualBlock(
                pose_prior_cf, NULL, node->pose().data());

        } else {
            ceres_problem_->SetParameterBlockConstant(node->pose().data());
        }

        solved_ = false;
        have_covariance_ = false;

        // Force the resulting object to be in front of this camera
        // ceres::CostFunction* front_cf =
        // FrontOfCameraConstraint::Create(body_T_camera_.rotation(),
        //                                                                 body_T_camera_.translation());
        // ceres_problem_->AddResidualBlock(
        //   front_cf, NULL, camera_poses_[pose_index].rotation_data(),
        //                   camera_poses_[pose_index].translation_data(),
        //                   object_pose_.translation_data());
    }
}

void
StructureOptimizationProblem::removeCamera(
  boost::shared_ptr<SemanticKeyframe> keyframe)
{
    auto existing_kf = keyframes_.find(keyframe->index());

    if (existing_kf == keyframes_.end()) {
        return;
    }

    auto node = local_pose_nodes_[keyframe->index()];

    keyframes_.erase(keyframe->index());
    local_pose_nodes_.erase(keyframe->index());

    ceres_problem_->RemoveParameterBlock(node->pose().data());

    if (prior_residual_ids_.find(keyframe->index()) !=
        prior_residual_ids_.end()) {
        ceres_problem_->RemoveResidualBlock(
          prior_residual_ids_[keyframe->index()]);
        prior_residual_ids_.erase(keyframe->index());
    }

    solved_ = false;
    have_covariance_ = false;
}

void
StructureOptimizationProblem::initializeKeypointPosition(
  size_t kp_id,
  const Eigen::Vector3d& p)
{
    *kps_[kp_id] = p;
}

void
StructureOptimizationProblem::initializePose(Pose3 pose)
{
    object_pose_ = pose;
}
