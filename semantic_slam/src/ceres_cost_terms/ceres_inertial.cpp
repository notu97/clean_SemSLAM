#include "semantic_slam/ceres_cost_terms/ceres_inertial.h"

#include "semantic_slam/inertial/InertialIntegrator.h"

InertialCostTerm::InertialCostTerm(
  double t0,
  double t1,
  boost::shared_ptr<InertialIntegrator> integrator)
  : t0_(t0)
  , t1_(t1)
  , integrator_(integrator)
  , have_preintegrated_(false)
{}

void
InertialCostTerm::preintegrate(const Eigen::VectorXd& bias0) const
{
    auto preint_xJP =
      integrator_->preintegrateInertialWithJacobianAndCovariance(
        t0_, t1_, bias0);

    preint_x_ = preint_xJP[0];
    preint_Jbias_ = preint_xJP[1];
    preint_P_ = preint_xJP[2];

    have_preintegrated_ = true;

    bias_at_integration_ = bias0;

    // std::cout << "PREINTEGRATION done\n";
    // std::cout << "preint x = " << preint_x_.transpose() << "\npreint P = \n"
    //           << preint_P_ << std::endl;

    gtsam_preintegrator_ = integrator_->createGtsamIntegrator(t0_, t1_, bias0);

    // preint_x_.head<4>() =
    //   gtsam_preintegrator_->deltaRij().toQuaternion().coeffs();
    // preint_x_.segment<3>(4) = gtsam_preintegrator_->deltaVij();
    // preint_x_.tail<3>() = gtsam_preintegrator_->deltaPij();

    // preint_P_.block<3, 3>(0, 0) =
    //   gtsam_preintegrator_->preintMeasCov().block<3, 3>(0, 0);
    // preint_P_.block<3, 3>(0, 3) =
    //   gtsam_preintegrator_->preintMeasCov().block<3, 3>(0, 6);
    // preint_P_.block<3, 3>(0, 6) =
    //   gtsam_preintegrator_->preintMeasCov().block<3, 3>(0, 3);
    // preint_P_.block<3, 3>(0, 9) =
    //   gtsam_preintegrator_->preintMeasCov().block<3, 3>(0, 12);
    // preint_P_.block<3, 3>(0, 12) =
    //   gtsam_preintegrator_->preintMeasCov().block<3, 3>(0, 9);

    // preint_P_.block<3, 3>(3, 0) =
    //   gtsam_preintegrator_->preintMeasCov().block<3, 3>(6, 0);
    // preint_P_.block<3, 3>(3, 3) =
    //   gtsam_preintegrator_->preintMeasCov().block<3, 3>(6, 6);
    // preint_P_.block<3, 3>(3, 6) =
    //   gtsam_preintegrator_->preintMeasCov().block<3, 3>(6, 3);
    // preint_P_.block<3, 3>(3, 9) =
    //   gtsam_preintegrator_->preintMeasCov().block<3, 3>(6, 12);
    // preint_P_.block<3, 3>(3, 12) =
    //   gtsam_preintegrator_->preintMeasCov().block<3, 3>(6, 9);

    // preint_P_.block<3, 3>(6, 0) =
    //   gtsam_preintegrator_->preintMeasCov().block<3, 3>(3, 0);
    // preint_P_.block<3, 3>(6, 3) =
    //   gtsam_preintegrator_->preintMeasCov().block<3, 3>(3, 6);
    // preint_P_.block<3, 3>(6, 6) =
    //   gtsam_preintegrator_->preintMeasCov().block<3, 3>(3, 3);
    // preint_P_.block<3, 3>(6, 9) =
    //   gtsam_preintegrator_->preintMeasCov().block<3, 3>(3, 12);
    // preint_P_.block<3, 3>(6, 12) =
    //   gtsam_preintegrator_->preintMeasCov().block<3, 3>(3, 9);

    // preint_P_.block<3, 3>(9, 0) =
    //   gtsam_preintegrator_->preintMeasCov().block<3, 3>(12, 0);
    // preint_P_.block<3, 3>(9, 3) =
    //   gtsam_preintegrator_->preintMeasCov().block<3, 3>(12, 6);
    // preint_P_.block<3, 3>(9, 6) =
    //   gtsam_preintegrator_->preintMeasCov().block<3, 3>(12, 3);
    // preint_P_.block<3, 3>(9, 9) =
    //   gtsam_preintegrator_->preintMeasCov().block<3, 3>(12, 12);
    // preint_P_.block<3, 3>(9, 12) =
    //   gtsam_preintegrator_->preintMeasCov().block<3, 3>(12, 9);

    // preint_P_.block<3, 3>(12, 0) =
    //   gtsam_preintegrator_->preintMeasCov().block<3, 3>(9, 0);
    // preint_P_.block<3, 3>(12, 3) =
    //   gtsam_preintegrator_->preintMeasCov().block<3, 3>(9, 6);
    // preint_P_.block<3, 3>(12, 6) =
    //   gtsam_preintegrator_->preintMeasCov().block<3, 3>(9, 3);
    // preint_P_.block<3, 3>(12, 9) =
    //   gtsam_preintegrator_->preintMeasCov().block<3, 3>(9, 12);
    // preint_P_.block<3, 3>(12, 12) =
    //   gtsam_preintegrator_->preintMeasCov().block<3, 3>(9, 9);
}

bool
InertialCostTerm::Evaluate(double const* const* parameters,
                           double* residuals_ptr,
                           double** jacobians) const
{
    Eigen::Map<const Eigen::VectorXd> qp0(parameters[0], 7);
    Eigen::Map<const Eigen::Vector3d> map_v_body0(parameters[1]);
    Eigen::Map<const Eigen::VectorXd> bias0(parameters[2], 6);

    Eigen::Map<const Eigen::VectorXd> qp1(parameters[3], 7);
    Eigen::Map<const Eigen::Vector3d> map_v_body1(parameters[4]);
    Eigen::Map<const Eigen::VectorXd> bias1(parameters[5], 6);

    Eigen::Map<const Eigen::Vector3d> gravity(parameters[6]);

    Eigen::Map<Eigen::Matrix<double, 15, 1>> residual(residuals_ptr);

    if (!have_preintegrated_)
        preintegrate(bias0);

    Pose3 pose0(qp0);
    Pose3 pose1(qp1);

    // Preintegration...
    // auto preint_xJP =
    //   integrator_->preintegrateInertialWithJacobianAndCovariance(
    //     t0_, t1_, bias0);

    // preint_x_ = preint_xJP[0];
    // preint_Jbias_ = preint_xJP[1];
    // preint_P_ = preint_xJP[2];

    Eigen::VectorXd delta_bias = bias0 - bias_at_integration_;

    if (delta_bias.norm() > 1e-4) {
        preintegrate(bias0);
        delta_bias.setZero();
    }

    // std::cout << "Delta bias = " << delta_bias.transpose() << std::endl;

    Eigen::VectorXd preint_effective = preint_x_ + preint_Jbias_ * delta_bias;

    Eigen::Quaterniond preint_q(preint_effective.head<4>());
    preint_q.normalize();

    double dt = t1_ - t0_;

    // Compute measurement model from given poses
    Eigen::Quaterniond q0 = pose0.rotation();
    q0.normalize();
    Eigen::Quaterniond q1 = pose1.rotation();
    q1.normalize();

    Eigen::Quaterniond q0_inv = q0.conjugate();
    Eigen::Quaterniond qpre_hat = q0_inv * q1;
    Eigen::Vector3d vpre_hat =
      q0_inv * (map_v_body1 - map_v_body0 - dt * gravity);
    Eigen::Vector3d ppre_hat =
      q0_inv * (pose1.translation() - pose0.translation() - map_v_body0 * dt -
                0.5 * gravity * dt * dt);

    // Compute residuals / errors between prehat and pre
    Eigen::Quaterniond qpre_hat_inv = qpre_hat.conjugate();
    Eigen::Quaterniond dq = preint_q * qpre_hat_inv;

    // residual ordering: [q v p bg ba]
    residual.head<3>() = 2 * dq.vec();
    residual.segment<3>(3) = vpre_hat - preint_effective.segment<3>(4);
    residual.segment<3>(6) = ppre_hat - preint_effective.tail<3>();
    residual.tail<6>() = bias1 - bias0;

    // Construct the full covariance of the residual
    // Right now P is covariance of [q v p] (in tangent space)
    // Eigen::MatrixXd Pfull = Eigen::MatrixXd::Zero(15, 15);
    // Pfull.block<9, 9>(0, 0) = preint_P_;
    // Pfull.block<6, 6>(9, 9) = dt * integrator_->randomWalkCovariance();

    Eigen::MatrixXd sqrtPinv = Eigen::MatrixXd::Identity(15, 15);
    preint_P_.llt().matrixL().solveInPlace(sqrtPinv);

    // std::cout << "Residual before P:\n"
    //           << residual.transpose() << "\nand P = \n"
    //           << Pfull << std::endl;

    residual.applyOnTheLeft(sqrtPinv);

    // std::cout << "Residual after P:\n" << residual.transpose() << std::endl;

    // Jacobians...
    if (jacobians != NULL) {
        using JacobianType = Eigen::Matrix<double, -1, -1, Eigen::RowMajor>;
        Eigen::Matrix3d Rq0_inv = q0_inv.toRotationMatrix();

        if (jacobians[0] != NULL) {
            // D(residuals)/D(qp0)
            Eigen::Map<JacobianType> J(jacobians[0], 15, 7);
            J.setZero();

            // First do d(dq)/dq0
            // dq = qpre * (q0^-1 * q1)^-1
            J.block<3, 4>(0, 0) =
              2.0 * (math::Dquat_mul_dq2(preint_q, qpre_hat_inv) *
                     math::Dquat_inv(qpre_hat) *
                     math::Dquat_mul_dq1(q0_inv, q1) * math::Dquat_inv(q0))
                      .topRows<3>();

            // d(dq) / dp0 = 0.

            // Next do d(dv) / d(q0)
            // dv = vpre_hat - vpre
            //    = q0_inv * (...) - vpre
            J.block<3, 4>(3, 0) = math::Dpoint_transform_transpose_dq(
              q0, map_v_body1 - map_v_body0 - dt * gravity);

            // d(dv) / dp0 = 0.

            // Next, d(dp) / (dq0).
            // Similar to d(dv) but with a different term multiplied
            J.block<3, 4>(6, 0) = math::Dpoint_transform_transpose_dq(
              q0,
              pose1.translation() - pose0.translation() - map_v_body0 * dt -
                0.5 * gravity * dt * dt);

            // Next, d(dp) / d(dp0)
            J.block<3, 3>(6, 4) = -Rq0_inv;

            // Bias terms d(b)/d(qp) = 0, so we're done.

            J.applyOnTheLeft(sqrtPinv);
        }

        if (jacobians[1] != NULL) {
            // Second parameter is velocity0. Only terms that depends on this is
            // the velocity & position pre residual.
            Eigen::Map<JacobianType> J(jacobians[1], 15, 3);
            J.setZero();

            // d(dv) / d(v0)
            J.block<3, 3>(3, 0) = -Rq0_inv;

            // d(dp) / d(v0)
            J.block<3, 3>(6, 0) = -dt * Rq0_inv;

            J.applyOnTheLeft(sqrtPinv);
        }

        if (jacobians[2] != NULL) {
            // Third parameter is bias0.
            Eigen::Map<JacobianType> J(jacobians[2], 15, 6);
            J.setZero();

            // First do d(dq)/d(bg0).
            // dq = qpre * (q0^-1 * q1)^-1, we have d(qpre)/d(bg0) from the
            // integrator
            J.block<3, 6>(0, 0) =
              2.0 * (math::Dquat_mul_dq1(preint_q, qpre_hat_inv) *
                     preint_Jbias_.block<4, 6>(0, 0))
                      .topRows<3>();

            // d(dv) and d(dp) are -Jbias
            J.block<6, 6>(3, 0) = -preint_Jbias_.block<6, 6>(4, 0);

            // d(bias0) is just -identity of course
            J.block<6, 6>(9, 0) = -Eigen::MatrixXd::Identity(6, 6);

            J.applyOnTheLeft(sqrtPinv);
        }

        if (jacobians[3] != NULL) {
            // Fourth parameter is qp1. Similar to qp0...
            Eigen::Map<JacobianType> J(jacobians[3], 15, 7);
            J.setZero();

            // First do d(dq)/dq1
            // dq = qpre * (q0^-1 * q1)^-1
            J.block<3, 4>(0, 0) =
              2.0 *
              (math::Dquat_mul_dq2(preint_q, qpre_hat_inv) *
               math::Dquat_inv(qpre_hat) * math::Dquat_mul_dq2(q0_inv, q1))
                .topRows<3>();

            // d(dq) / dp1 = 0.

            // Unlike the q0 case, here d(dv)/d(q1) = d(dp)/d(q1) = 0.

            // Only remaining nonzero Jacobian is d(dp)/d(p1).
            J.block<3, 3>(6, 4) = Rq0_inv;

            J.applyOnTheLeft(sqrtPinv);
        }

        if (jacobians[4] != NULL) {
            // Fifth parameter is v1.
            Eigen::Map<JacobianType> J(jacobians[4], 15, 3);
            J.setZero();

            // Here the only dependent residual is d(dv)/d(v1)
            J.block<3, 3>(3, 0) = Rq0_inv;

            J.applyOnTheLeft(sqrtPinv);
        }

        if (jacobians[5] != NULL) {
            // d/dbias1.
            Eigen::Map<JacobianType> J(jacobians[5], 15, 6);
            J.setZero();

            // Only thing that matters here is the bias residual
            J.block<6, 6>(9, 0) = Eigen::MatrixXd::Identity(6, 6);

            J.applyOnTheLeft(sqrtPinv);
        }

        if (jacobians[6] != NULL) {
            // Finally the jacobians w.r.t. gravity.
            // Only affects d(dv) and d(dp)
            Eigen::Map<JacobianType> J(jacobians[6], 15, 3);
            J.setZero();

            J.block<3, 3>(3, 0) = -dt * Rq0_inv;
            J.block<3, 3>(6, 0) = -0.5 * dt * dt * Rq0_inv;

            J.applyOnTheLeft(sqrtPinv);
        }
    }

    return true;
}

ceres::CostFunction*
InertialCostTerm::Create(double t0,
                         double t1,
                         boost::shared_ptr<InertialIntegrator> integrator)
{
    return new InertialCostTerm(t0, t1, integrator);
}