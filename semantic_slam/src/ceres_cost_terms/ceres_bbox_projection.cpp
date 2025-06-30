
#include "semantic_slam/ceres_cost_terms/ceres_bbox_projection.h"

ProjectionCostTermBbox::ProjectionCostTermBbox(
  const Eigen::Vector3d& measured_bline,
  const Eigen::Vector3d& object_shape,
  const Eigen::Quaterniond& body_q_camera,
  const Eigen::Vector3d& body_p_camera,
  const Eigen::Matrix3d& camera_intrinsic)
  : measured_bline_(measured_bline)
  , object_shape_(object_shape)
  , body_q_camera_(body_q_camera)
  , body_p_camera_(body_p_camera)
  , camera_intrinsic_(camera_intrinsic)
{
  
  Eigen::Vector3d vsq = ((object_shape.array())/2).square().matrix();
  std::cout << "vsq: " << vsq(0) << " " << vsq(1) << " " << vsq(2) << std::endl;
  // vsq = vsq.array() / 2;
  // std::cout << "vsq after: " << vsq(0) << " " << vsq(1) << " " << vsq(2) << std::endl;
  Eigen::Vector4d vsqh;
  vsqh << vsq, -1;
  Qi_ = vsqh.asDiagonal();

}

template<typename T>
bool
ProjectionCostTermBbox::operator()(const T* const map_x_body_ptr,
                                   const T* const map_x_obj_ptr,
                                   T* residual_ptr) const
{
    // static_assert(sizeof(T) != sizeof(double));
    Eigen::Map<const Eigen::Quaternion<T>> map_q_body(map_x_body_ptr);
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> map_p_body(map_x_body_ptr + 4);
    
    Eigen::Map<const Eigen::Quaternion<T>> map_q_obj(map_x_obj_ptr);
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> map_p_obj(map_x_obj_ptr + 4);

    Eigen::Matrix<T, 4, 4> wTi;
    wTi.template block<3, 3>(0, 0) = map_q_body.toRotationMatrix();
    wTi.template block<3, 1>(0, 3) = map_p_body;
    wTi.template block<1, 4>(3, 0) << T(0), T(0), T(0), T(1);

    Eigen::Matrix<T, 4, 4> iTc;
    iTc.template block<3, 3>(0, 0) = body_q_camera_.cast<T>().toRotationMatrix();
    iTc.template block<3, 1>(0, 3) = body_p_camera_.cast<T>();
    iTc.template block<1, 4>(3, 0) << T(0), T(0), T(0), T(1);

    Eigen::Matrix<T, 4, 4> wTo;
    wTo.template block<3, 3>(0, 0) = map_q_obj.toRotationMatrix();
    wTo.template block<3, 1>(0, 3) = map_p_obj;
    wTo.template block<1, 4>(3, 0) << T(0), T(0), T(0), T(1);

    Eigen::Matrix<T, 4, 4> cTo = iTc.inverse() * wTi.inverse() * wTo;

    Eigen::Matrix<T, 3, 4> P = camera_intrinsic_.cast<T>() * cTo.topRows(3);
    Eigen::Matrix<T, 3, 3> U_square = (Qi_.block<3, 3>(0, 0)).cast<T>();

    Eigen::Matrix<T, 4, 1> uline_b = P.transpose() * measured_bline_.cast<T>();
    Eigen::Matrix<T, 3, 1> b = uline_b.template block<3, 1>(0, 0);
    T b_norm = b.norm();

    // note, plane_orig_dist is -bh in paper 
    T plane_orig_dist = uline_b(3, 0);
    Eigen::Matrix<T, 1, 1> sqrt_bU2b = Eigen::Matrix<T, 1, 1>((b.transpose() * U_square * b).array().sqrt());
    T sign = T(plane_orig_dist > 0 ? 1.0 : -1.0);

    // T error[] = {Eigen::Matrix<T, 1, 1>(plane_orig_dist - sign * sqrt_bU2b.array()).coeffRef(0) / b_norm};
    // Eigen::Map<const Eigen::Matrix<T, -1, 1>> resi(error, 1);

    // Compute residual
    Eigen::Map<Eigen::Matrix<T, 1, 1>> residual(residual_ptr);
    residual = (plane_orig_dist - sign * sqrt_bU2b.array()) / b_norm;
    // residuals.template segment<1>(0) = resi;
    // residual = sqrt_information_ * (measured_ - zhat);

    return true;
}

ceres::CostFunction*
ProjectionCostTermBbox::Create(
  const Eigen::Vector3d& measured_bline,
  const Eigen::Vector3d& object_shape,
  const Eigen::Quaterniond& body_q_camera,
  const Eigen::Vector3d& body_p_camera,
  const Eigen::Matrix3d& camera_intrinsic)
{
    return new ceres::AutoDiffCostFunction<ProjectionCostTermBbox, 1, 7, 7>(
      new ProjectionCostTermBbox(measured_bline,
                                 object_shape,
                                 body_q_camera,
                                 body_p_camera,
                                 camera_intrinsic));
}