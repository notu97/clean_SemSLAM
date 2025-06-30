#pragma once

#include "semantic_slam/Pose3.h"

#include <ceres/autodiff_cost_function.h>

#include "semantic_slam/CameraCalibration.h"

class ProjectionCostTermBbox
{
  public:
    ProjectionCostTermBbox(const Eigen::Vector3d& measured_bbox,
                       const Eigen::Vector3d& object_shape,
                       const Eigen::Quaterniond& body_q_camera,
                       const Eigen::Vector3d& body_p_camera,
                       const Eigen::Matrix3d& camera_intrinsic);

    template<typename T>
    bool operator()(const T* const map_x_body_ptr,
                    const T* const map_x_obj_ptr,
                    T* residual_ptr) const;

    static ceres::CostFunction* Create(
      const Eigen::Vector3d& measured,
      const Eigen::Vector3d& object_shape,
      const Eigen::Quaterniond& body_q_camera,
      const Eigen::Vector3d& body_p_camera,
      const Eigen::Matrix3d& camera_intrinsic);

    void setCameraCalibration(double fx, double fy, double cx, double cy);

  private:
    Eigen::Vector3d measured_bline_;
    Eigen::Vector3d object_shape_;
    Eigen::Matrix4d Qi_;
    Eigen::Quaterniond body_q_camera_;
    Eigen::Vector3d body_p_camera_;
    Eigen::Matrix3d camera_intrinsic_;

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};
