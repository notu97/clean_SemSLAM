#include "semantic_slam/Camera.h"

#include "semantic_slam/CameraCalibration.h"

Camera::Camera(Pose3 pose, boost::shared_ptr<CameraCalibration> calibration)
  : pose_(pose)
  , calibration_(calibration)
{}

Eigen::Vector2d
Camera::project(const Eigen::Vector3d& p,
                boost::optional<Eigen::MatrixXd&> Hpose,
                boost::optional<Eigen::MatrixXd&> Hpoint) const
{
    // Convert to camera coordinates
    Eigen::MatrixXd Htransform_pose, Htransform_point;
    Eigen::Vector3d C_p;
    // std::cout << "project 1" << std::endl;
    if (Hpose || Hpoint)
        C_p = pose_.transform_to(p, Htransform_pose, Htransform_point);
    else
        C_p = pose_.transform_to(p);

    if (C_p(2) <= 0) {
        throw CheiralityException("Cheirality exception");
    }
    // std::cout << "project 2" << std::endl;
    // Project to normalized coordinates
    double zinv = 1.0 / C_p(2);
    Eigen::Vector2d xy(zinv * C_p(0), zinv * C_p(1));

    Eigen::Matrix<double, 2, 3> Hproject;

    if (Hpose || Hpoint) {
        Hproject << 1, 0, -C_p(0) / C_p(2), 0, 1, -C_p(1) / C_p(2);
        Hproject /= C_p(2);
    }
    // std::cout << "project 3" << std::endl;
    // Eigen::Matrix<double, 2, 3> Dproject;
    // Dproject << zinv, 0.0, -C_p(0) * zinv, 0.0, zinv, -C_p(1) * zinv;

    // Distort if we have a calibration
    Eigen::MatrixXd Hcalib = Eigen::MatrixXd::Identity(2, 2);
    // Eigen::MatrixXd Hcalib;
    // Hcalib(0, 0) = calibration_->fx();
    // Hcalib(0, 1) = calibration_->skew();
    // Hcalib(1, 1) = calibration_->fy();

    if (calibration_) {
        // std::cout << "Has calibration." << std::endl;
        if (Hpose || Hpoint){
            // std::cout << "project 4" << std::endl;
            xy = calibration_->uncalibrate(xy, Hcalib);
            // std::cout << "project 5" << std::endl;
        }
        else
            xy = calibration_->uncalibrate(xy);
    }
    
    // std::cout << "Hcalib:\n" << Hcalib << std::endl;
    if (Hpose) {
        *Hpose = Hcalib * Hproject * Htransform_pose;
    }
    
    if (Hpoint) {
        *Hpoint = Hcalib * Hproject * Htransform_point;
    }
    
    return xy;
}

Eigen::Vector2d
Camera::calibrate(const Eigen::Vector2d& p) const
{
    if (calibration_)
        return calibration_->calibrate(p);
    else
        return p;
}

Eigen::Vector2d
Camera::uncalibrate(const Eigen::Vector2d& p) const
{
    if (calibration_)
        return calibration_->uncalibrate(p);
    else
        return p;
}