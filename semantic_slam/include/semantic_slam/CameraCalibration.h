#pragma once

#include <boost/optional.hpp>
#include <eigen3/Eigen/Core>

#include <gtsam/geometry/Cal3DS2.h>

// Camera calibration data & utilities

class CameraCalibration
{
  public:
    CameraCalibration() {}

    CameraCalibration(double fx,
                      double fy,
                      double s,
                      double u0,
                      double v0,
                      double k1,
                      double k2,
                      double p1 = 0.0,
                      double p2 = 0.0);

  private:
    double fx_, fy_, s_, u0_, v0_;
    double k1_, k2_; // radial distortion
    double p1_, p2_; // tangential distortion

  public:
    inline double fx() const { return fx_; }
    inline double fy() const { return fy_; }
    inline double skew() const { return s_; }
    inline double u0() const { return u0_; }
    inline double v0() const { return v0_; }
    inline double k1() const { return k1_; }
    inline double k2() const { return k2_; }
    inline double p1() const { return p1_; }
    inline double p2() const { return p2_; }

    // Convert intrinsic camera coordinates to distorted image coordinates
    // Eigen::Vector2d uncalibrate(const Eigen::Vector2d& p,
    //                             boost::optional<Eigen::MatrixXd&> Hpoint =
    //                             boost::none) const;

    template<typename T>
    Eigen::Matrix<T, 2, 1> uncalibrate(
      const Eigen::Matrix<T, 2, 1>& p,
      boost::optional<Eigen::MatrixXd&> Hpoint = boost::none) const;

    void Duncalibrate(const Eigen::Vector2d& p, Eigen::MatrixXd& Hpoint) const;

    // Convert distorted image coordinates to intrinsic coordinates
    Eigen::Vector2d calibrate(const Eigen::Vector2d& p) const;

    operator gtsam::Cal3DS2() const;
};

template<typename T>
Eigen::Matrix<T, 2, 1>
CameraCalibration::uncalibrate(const Eigen::Matrix<T, 2, 1>& p,
                               boost::optional<Eigen::MatrixXd&> Hpoint) const
{
    // Code borrowed from opencv's projectpoints function
    // std::cout << "uncalibrateh" << std::endl;
    T x = p(0), y = p(1);
    T r2 = x * x + y * y;
    T r4 = r2 * r2;
    // std::cout << "uncalibrate 1" << std::endl;
    T a1 = T(2.0) * x * y;
    T a2 = r2 + T(2.0) * x * x;
    T a3 = r2 + T(2.0) * y * y;
    // std::cout << "uncalibrate 1" << std::endl;
    T cdist = T(1.0) + k1_ * r2 + k2_ * r4;

    T xd0 = x * cdist + p1_ * a1 + p2_ * a2;
    T yd0 = y * cdist + p1_ * a3 + p2_ * a1;

    if (Hpoint) {
        throw std::runtime_error(
          "Error: requesting Hpoint with incompatible data type");
    }

    // if (Hpoint) {
    //   *Hpoint << fx_, s_, 0, fy_;
    // }
    return Eigen::Matrix<T, 2, 1>(fx_ * xd0 + s_ * yd0 + u0_, fy_ * yd0 + v0_);
}