#include "semantic_slam/SmartProjectionFactor.h"

#include "semantic_slam/CameraCalibration.h"
#include "semantic_slam/CameraSet.h"
#include "semantic_slam/CeresProjectionFactor.h"

#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/slam/SmartProjectionPoseFactor.h>

SmartProjectionFactor::SmartProjectionFactor(
  const Pose3& body_T_sensor,
  boost::shared_ptr<CameraCalibration> calibration,
  double reprojection_error_threshold,
  int tag)
  : CeresFactor(FactorType::SMART_PROJECTION, tag)
  , I_T_C_(body_T_sensor)
  , calibration_(calibration)
  , reprojection_error_threshold_(reprojection_error_threshold)
  , in_graph_(false)
  , triangulation_good_(false)
{
    // Parameter block ordering:
    // camera poses (q1 p1), (q2 p2), ...
    // [q1 p1 q2 p2 ... qn pn]

    // TODO get the real value
    auto gtsam_noise = gtsam::noiseModel::Isotropic::Sigma(2, 4);

    gtsam::SmartProjectionParams projection_params;
    projection_params.degeneracyMode =
      gtsam::DegeneracyMode::ZERO_ON_DEGENERACY;
    projection_params.linearizationMode = gtsam::LinearizationMode::HESSIAN;
    projection_params.setLandmarkDistanceThreshold(1e6);
    projection_params.setRankTolerance(1e-2);
    projection_params.triangulation.dynamicOutlierRejectionThreshold =
      reprojection_error_threshold;

    // This factor will change as we add measurements.
    // GTSAM's isam implementation cannot handle factors changing after they've
    // been added to the ISAM structure, so we can't simply allocate a factor
    // here and modify it. Instead we'll allocate a new one each time we receive
    // a measurement

    // gtsam_factor_ = util::allocate_aligned<GtsamFactorType>(gtsam_noise,
    //                                                         util::allocate_aligned<gtsam::Cal3DS2>(*calibration),
    //                                                         gtsam::Pose3(body_T_sensor),
    //                                                         projection_params);
}

CeresFactor::Ptr
SmartProjectionFactor::clone() const
{
    auto fac = util::allocate_aligned<SmartProjectionFactor>(
      I_T_C_, calibration_, reprojection_error_threshold_, tag_);

    for (size_t i = 0; i < msmts_.size(); ++i) {
        fac->addMeasurement(nullptr, msmts_[i], covariances_[i]);
    }

    return fac;
}

size_t
SmartProjectionFactor::nMeasurements() const
{
    return msmts_.size();
}

bool
SmartProjectionFactor::decideIfTriangulate(
  const aligned_vector<Pose3>& body_poses) const
{
    // Triangulate if:
    //  (1) the last triangulation failed,
    //  (2) camera poses are sufficiently different from last time,
    //  (3) or, the number of camera poses is different.
    if (!triangulation_good_ ||
        body_poses.size() != triangulation_poses_.size()) {
        return true;
    }

    // Compare each camera pose
    bool equal = true;
    for (size_t i = 0; i < body_poses.size(); ++i) {
        Pose3 camera_pose = body_poses[i].compose(I_T_C_);

        if (!camera_pose.equals(triangulation_poses_[i], 1e-5)) {
            equal = false;
            break;
        }
    }

    return !equal;
}

void
SmartProjectionFactor::triangulate(
  const aligned_vector<Pose3>& body_poses) const
{
    triangulation_good_ = false;
    triangulation_poses_.clear();

    CameraSet cameras;
    for (size_t i = 0; i < msmts_.size(); ++i) {
        triangulation_poses_.push_back(body_poses[i].compose(I_T_C_));
        Camera camera(triangulation_poses_[i], calibration_);
        cameras.addCamera(camera);
    }

    // double cond;
    TriangulationResult triangulation =
      cameras.triangulateMeasurementsApproximate(msmts_, 10);

    if (triangulation.max_reprojection_error <= reprojection_error_threshold_ &&
        triangulation.status == TriangulationStatus::SUCCESS) {
        landmark_position_ = triangulation.point;
        triangulation_good_ = true;
    }

    // std::cout << "Triangulation = " << triangulation.point.transpose() <<
    // std::endl;
}

void
SmartProjectionFactor::addToProblem(boost::shared_ptr<ceres::Problem> problem)
{
    // TODO add huber loss
    in_graph_ = true;

    aligned_vector<Pose3> body_poses;
    for (size_t i = 0; i < msmts_.size(); ++i) {
        body_poses.push_back(camera_node(i)->pose());
    }

    triangulate(body_poses);

    auto problem_it = std::find(problems_.begin(), problems_.end(), problem);
    if (problem_it == problems_.end()) {
        problems_.push_back(problem);
        internalAddToProblem(problem);
    } else {
        // ROS_WARN("Tried to add a factor to a problem it's already in.");
    }
}

void
SmartProjectionFactor::removeFromProblem(
  boost::shared_ptr<ceres::Problem> problem)
{
    auto problem_it = std::find(problems_.begin(), problems_.end(), problem);
    if (problem_it != problems_.end()) {
        problems_.erase(problem_it);
        internalRemoveFromProblem(problem);
    } else {
        // ROS_WARN("Tried to remove a factor from a problem it's not in");
    }
}

void
SmartProjectionFactor::internalAddToProblem(
  boost::shared_ptr<ceres::Problem> problem)
{
    // Set up parameter block sizes and pointers
    mutable_parameter_block_sizes()->clear();
    parameter_blocks_.clear();

    for (size_t i = 0; i < nodes_.size(); ++i) {
        mutable_parameter_block_sizes()->push_back(7);
        parameter_blocks_.push_back(camera_node(i)->pose().data());
    }

    set_num_residuals(2 * nMeasurements() - 3);
    // std::cout << "SmartProjectionFactor cost function block size: " << this->parameter_block_sizes().size() << std::endl;
    // std::cout << "SmartProjectionFactor parameter_blocks_ size: " << parameter_blocks_.size() << std::endl;
    if (this->parameter_block_sizes().size() != parameter_blocks_.size()) std::cout << "SmartProjectionFactor unequal" << std::endl;
    // if (this->parameter_block_sizes().size() == 19) std::cout << "SmartProjectionFactor cost function block size equals19" << std::endl;
    // if (this->parameter_block_sizes().size() == 10) std::cout << "SmartProjectionFactor cost function block size equals10" << std::endl;
    ceres::ResidualBlockId residual_id =
      problem->AddResidualBlock(this, NULL, parameter_blocks_);
    residual_ids_[problem.get()] = residual_id;
    active_ = true;
}

void
SmartProjectionFactor::internalRemoveFromProblem(
  boost::shared_ptr<ceres::Problem> problem)
{
    auto it = residual_ids_.find(problem.get());
    if (it != residual_ids_.end()) {
        problem->RemoveResidualBlock(it->second);
        residual_ids_.erase(it);
    }

    active_ = !residual_ids_.empty();
}

void
SmartProjectionFactor::addMeasurement(SE3NodePtr body_pose_node,
                                      const Eigen::Vector2d& pixel_coords,
                                      const Eigen::Matrix2d& msmt_covariance)
{
    // Compute the reprojection error...
    // if (in_graph_) {
    //     try {
    //         Camera camera(body_pose_node->pose().compose(I_T_C_),
    //         calibration_); Eigen::Vector2d zhat =
    //         camera.project(landmark_position_); double error = (pixel_coords
    //         - zhat).transpose() * msmt_covariance.llt().solve(pixel_coords -
    //         zhat);

    //         if (error > chi2inv99(2) || !std::isfinite(error)) return;

    //         // ROS_INFO_STREAM("[SmartProjectionFactor] Added measurement
    //         with reprojection error " << error);
    //         // std::cout << " zhat = " << zhat.transpose() << " ; msmt = " <<
    //         pixel_coords.transpose() << std::endl;
    //     } catch (CheiralityException& e) {
    //         return;
    //         // ROS_INFO_STREAM("[SmartProjectionFactor] Added measurement
    //         behind camera!");
    //     }
    // }

    // body_poses_.push_back(body_pose_node);
    msmts_.push_back(pixel_coords);
    nodes_.push_back(body_pose_node);
    covariances_.push_back(msmt_covariance);

    sqrt_informations_.push_back(Eigen::Matrix2d::Identity());
    Eigen::Matrix2d sqrtC = msmt_covariance.llt().matrixL();
    sqrtC.triangularView<Eigen::Lower>().solveInPlace(
      sqrt_informations_.back());

    // Ceres can't handle block sizes changing if we've already been added to
    // the Problem. So we need to remove and re-add ourself.
    // Updating block sizes in the necessary ceres::CostFunction-inherited
    // values is done in internalAddToProblem.

    // it's tempting to combine the following loops into one but we CANNOT.
    // internalAddToProblem modifies the residual/parameter structure exposed to
    // ceres.
    // TODO fix this unexpected behavior?
    if (in_graph_) {
        for (auto& problem : problems_) {
            internalRemoveFromProblem(problem);
        }

        for (auto& problem : problems_) {
            internalAddToProblem(problem);
        }
    }

    // aligned_vector<Pose3> body_poses;
    // for (int i = 0; i < msmts_.size(); ++i) {
    //     if (camera_node(i))
    //         body_poses.push_back(camera_node(i)->pose());
    // }

    // triangulate(body_poses);

    // if (in_graph_) {
    //     for (auto& problem : problems_)
    //         internalAddToProblem(problem);
    // }

    // We need to recreate the factor each time a new measurement is added;
    // cannot simply update our old one.
    createGtsamFactor();
}

bool
SmartProjectionFactor::Evaluate(double const* const* parameters,
                                double* residuals_ptr,
                                double** jacobians) const
{
    // std::cout << "SmartProjectionFactor Evaluate" << std::endl;
    // Collect parameters
    aligned_vector<Pose3> body_poses;
    for (size_t i = 0; i < nMeasurements(); ++i) {
        Eigen::Map<const Eigen::VectorXd> qp(parameters[i], 7);

        body_poses.push_back(Pose3(qp));
    }

    Eigen::Map<Eigen::VectorXd> residuals(residuals_ptr, num_residuals());

    using JacobianType =
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    if (decideIfTriangulate(body_poses)) {
        triangulate(body_poses);
    }

    // If the triangulation was good, compute & use the actual costs and
    // Jacobians. If it wasn't, zero out the residuals and jacobian, which is
    // equivalent to not including this factor in the estimation
    if (!triangulation_good_) {
        residuals.setZero();
        if (jacobians) {
            for (size_t i = 0; i < nMeasurements(); ++i) {
                if (jacobians[i]) {
                    Eigen::Map<JacobianType> Dr_dx(
                      jacobians[i], num_residuals(), 7);
                    Dr_dx.setZero();
                }
            }
        }

        return true;
    }

    // Iterate through each measurement computing its residual & jacobian
    // If we need the Jacobians, compute them as we go...
    // Indices corresponding to this are 0 (pt), 2*i, and 1 + 2*i...

    // Begin by collecting all the jacobians
    // We have to do this whether ceres is requesting the Jacobians or not
    // because we need to project the residual into the point jacobian's null
    // space
    // std::cout << "Hpoint 1" << std::endl;
    Eigen::MatrixXd Hpoint = Eigen::MatrixXd::Zero(2 * nMeasurements(), 3);
    
    Eigen::MatrixXd Hpose =
      Eigen::MatrixXd::Zero(2 * nMeasurements(), 7 * nMeasurements());
    
    Eigen::VectorXd full_residual = Eigen::VectorXd::Zero(2 * nMeasurements());
    
    for (size_t i = 0; i < nMeasurements(); ++i) {
        Eigen::MatrixXd Hpose_compose;
        
        Pose3 G_T_C = body_poses[i].compose(I_T_C_, Hpose_compose, boost::none);

        Camera cam(G_T_C, calibration_);

        Eigen::MatrixXd Hpose_project, Hpoint_project;
        Eigen::Vector2d zhat;
        
        try {
            zhat =
              cam.project(landmark_position_, Hpose_project, Hpoint_project);
            // std::cout << "Hpoint 2" << std::endl;
            Hpose.block<2, 7>(2 * i, 7 * i) =
              -sqrt_informations_[i] * Hpose_project * Hpose_compose;
            
            Hpoint.block<2, 3>(2 * i, 0) =
              -sqrt_informations_[i] * Hpoint_project;

            full_residual.segment<2>(2 * i) =
              sqrt_informations_[i] * (msmts_[i] - zhat);
        } catch (CheiralityException& e) {
            // ignore this measurement for now
            // jacobians, residual already zeroed
        }
    }

    // Jacobian matrices are filled in
    // Compute the basis of Hpoint's left null space

    // Eigen::HouseholderQR<Eigen::MatrixXd> qr(Hpoint);
    // Eigen::MatrixXd Q = qr.householderQ();
    // Eigen::MatrixXd basis = Q.rightCols(num_residuals());
    // std::cout << "Hpoint 3" << std::endl;
    Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> cod(
      Hpoint.transpose());
    Eigen::MatrixXd V = cod.matrixZ().transpose();
    // Eigen::MatrixXd basis = V.block(0, cod.rank(), V.rows(), V.cols() -
    // cod.rank());
    Eigen::MatrixXd basis = V.block(0, 3, V.rows(), num_residuals());
    basis.applyOnTheLeft(cod.colsPermutation());

    Hpose.applyOnTheLeft(basis.transpose());
    // full_residual.applyOnTheLeft(basis.transpose());

    // Fill in Ceres jacobian data pointers
    if (jacobians) {
        for (size_t i = 0; i < nMeasurements(); ++i) {
            if (jacobians[i]) {
                Eigen::Map<JacobianType> Dr_dx(
                  jacobians[i], num_residuals(), 7);

                Dr_dx = Hpose.block(0, 7 * i, num_residuals(), 7);
            }
            // if (jacobians[2 * i]) {
            //     Eigen::Map<JacobianType> Dr_dq(
            //       jacobians[2 * i], num_residuals(), 4);
            //     Dr_dq = Hpose.block(0, 7 * i, num_residuals(), 4);
            // }

            // if (jacobians[2 * i + 1]) {
            //     Eigen::Map<JacobianType> Dr_dp(
            //       jacobians[2 * i + 1], num_residuals(), 3);
            //     Dr_dp = Hpose.block(0, 7 * i + 4, num_residuals(), 3);
            // }
        }
    }

    residuals = basis.transpose() * full_residual;
    // std::cout << "Hpoint 4" << std::endl;
    return true;
}

void
SmartProjectionFactor::createGtsamFactor() const
{
    if (msmts_.size() == 0)
        return;
    if (!camera_node(0))
        return;

    // have to assume that all measurements have the same covariance
    auto gtsam_noise = gtsam::noiseModel::Gaussian::Covariance(covariances_[0]);

    gtsam::SmartProjectionParams projection_params;
    projection_params.degeneracyMode =
      gtsam::DegeneracyMode::ZERO_ON_DEGENERACY;
    projection_params.linearizationMode = gtsam::LinearizationMode::HESSIAN;
    projection_params.setLandmarkDistanceThreshold(1e6);
    projection_params.setRankTolerance(1e-2);
    projection_params.triangulation.dynamicOutlierRejectionThreshold =
      reprojection_error_threshold_;

    // our calibration == nullptr corresponds to an already calibrated
    // camera, i.e. cx = cy = 0 and fx = fy = 1, which is what the default gtsam
    // calibration constructor provides
    auto gtsam_calib = calibration_
                         ? util::allocate_aligned<gtsam::Cal3DS2>(*calibration_)
                         : util::allocate_aligned<gtsam::Cal3DS2>();

    gtsam_factor_ = util::allocate_aligned<GtsamFactorType>(
      gtsam_noise, gtsam_calib, gtsam::Pose3(I_T_C_), projection_params);

    for (size_t i = 0; i < msmts_.size(); ++i) {
        gtsam_factor_->add(msmts_[i], camera_node(i)->key());
    }
}

void
SmartProjectionFactor::addToGtsamGraph(
  boost::shared_ptr<gtsam::NonlinearFactorGraph> graph) const
{
    if (!gtsam_factor_)
        createGtsamFactor();

    graph->push_back(gtsam_factor_);
}