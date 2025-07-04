// math utilities

#pragma once

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>
#include <boost/array.hpp>
#include <chrono>

// #include <ros/ros.h>

#include <boost/optional.hpp>
#include <boost/preprocessor/arithmetic/sub.hpp>
// #include <geometry_msgs/PoseWithCovarianceStamped.h>
// #include <sensor_msgs/Imu.h>

#include "semantic_slam/Pose3.h"

#include <chrono>
using TimePoint = std::chrono::steady_clock::time_point;

#define CONCAT_2(A, B) A##B
#define CONCAT_1(A, B) CONCAT_2(A, B)
#define TIMER_NAME_GEN_INNER(num) CONCAT_1(_semslam_tictoc_t1_, num)
#define TIMER_NAME_GEN TIMER_NAME_GEN_INNER(__COUNTER__)

#define TIMER_NAME_REF TIMER_NAME_GEN_INNER(BOOST_PP_SUB(__COUNTER__, 1))

#define TIME_THIS(x, name)                                                     \
    do {                                                                       \
        std::chrono::high_resolution_clock::time_point _semslam_timethis_t1 =  \
          std::chrono::high_resolution_clock::now();                           \
        x;                                                                     \
        std::chrono::high_resolution_clock::time_point _semslam_timethis_t2 =  \
          std::chrono::high_resolution_clock::now();                           \
        auto _semslam_timethis_duration =                                      \
          std::chrono::duration_cast<std::chrono::milliseconds>(               \
            _semslam_timethis_t2 - _semslam_timethis_t1)                       \
            .count();                                                          \
        ROS_INFO_STREAM(name << " took " << _semslam_timethis_duration         \
                             << " milliseconds.");                             \
    } while (0)

#define TIME_TIC                                                               \
    std::chrono::high_resolution_clock::time_point TIMER_NAME_GEN =            \
      std::chrono::high_resolution_clock::now()

#define TIME_TOC                                                               \
    ([&] {                                                                     \
        std::chrono::duration<double, std::micro> _semslam_tictoc_duration =   \
          std::chrono::high_resolution_clock::now() - TIMER_NAME_REF;          \
        return _semslam_tictoc_duration.count() / 1000.0;                      \
    })()

namespace util {

// Allocate a boost::shared_ptr to an object using Eigen's aligned allocator
// Saves a bit of typing
template<typename T, typename... Args>
boost::shared_ptr<T>
allocate_aligned(Args&&... args)
{
    return boost::allocate_shared<T>(Eigen::aligned_allocator<T>(),
                                     std::forward<Args>(args)...);
}

} // namespace util

// inline ros::Duration
// abs_duration(ros::Duration d)
// {
//     if (d.sec < 0 || (d.sec == 0 && d.nsec < 0)) {
//         return -d;
//     } else {
//         return d;
//     }
// }

// void appendToValues(gtsam::Values& v, const gtsam::Values& v_other) {
//   for(gtsam::Values::const_iterator key_value = v_other.begin(); key_value !=
//   v_other.end(); ++key_value) {
//     if (v.exists(key_value->key)) v.update(key_value->key, key_value->value);
//     else v.insert(key_value->key, key_value->value);
//   }
// }

template<typename T>
T
computeMedian(std::vector<T> vec)
{
    T median;
    size_t size = vec.size();

    std::sort(vec.begin(), vec.end());

    if (size % 2 == 0) {
        median = (vec[size / 2 - 1] + vec[size / 2]) / 2;
    } else {
        median = vec[size / 2];
    }

    return median;
}

/**
 * Computes the jacobians of the projection of a global point onto a camera.
 * Inputs:
 *   - G_R_I rotation from Global frame to IMU frame
 *   - G_t_I translation from global to IMU frame
 *   - I_R_C rotation from IMU to camera frame
 *   - G_l   3d position of projected point
 *
 * Returns:
 *   - H is a 2 by 9 jacobian matrix = [Hl Hq Hp]
 */
Eigen::Matrix<double, 2, 9>
computeProjectionJacobian(const Pose3& G_T_I,
                          const Pose3& I_T_C,
                          const Eigen::Vector3d& G_l);

template<typename Derived>
inline Eigen::Matrix<typename Derived::Scalar, 3, 3>
skewsymm(const Eigen::MatrixBase<Derived>& x)
{
    using T = typename Derived::Scalar;
    Eigen::Matrix<T, 3, 3> S;
    S << T(0.0), -x(2), x(1), x(2), T(0.0), -x(0), -x(1), x(0), T(0.0);
    return S;
}

/**
 * For any given angle theta \in R returns the equivalent rotation theta' \in
 * (-pi, pi]
 */
double
clamp_angle(double angle);

/**
 * Returns the rotation matrix R that minimizes the error S1 - R*S2
 * S1, S2 are 3xN matrices of corresponding points
 */
Eigen::Matrix3d
findRotation(const Eigen::MatrixXd& S1, const Eigen::MatrixXd& S2);

/**
 * Convert chrono::system_clock::time_point to double
 */
double timePointToSeconds(const boost::optional<std::chrono::steady_clock::time_point>& tpOpt);

/**
 * Aligns the points S2 (3xN) to S1 (3xN)
 * If w provided as an output, finds a full similarity transform including
 * scaling. If no w output provided, only computes rotation and translation.
 */
void
findSimilarityTransform(const Eigen::MatrixXd& S1,
                        const Eigen::MatrixXd& S2,
                        Eigen::Matrix3d& R,
                        Eigen::Vector3d& T,
                        boost::optional<double&> w = boost::none);

// approx chi2inv(.95, dof)
inline double
chi2inv95(int dof)
{
    if (dof == 2)
        return 5.991464547107981;
    else
        return (-0.0045644 * dof + 1.4756330) * dof + 3.7616227;
}

// approx chi2inv(.99, dof)
inline double
chi2inv99(int dof)
{
    if (dof == 2)
        return 9.210340371976180;
    else
        return (-0.0065001 * dof + 1.6746717) * dof + 6.7343300;
}

// The (squared) mahalanobis distance with n d.o.f. would ideally be compared to
// some threshold drawn from the Chi2(n) cdf. If we need to compare multiple
// such distances to a single threshold (e.g. in munkres) we can instead *scale*
// them so comparing them to Chi2(2) produces the same result. This function
// returns the (approximate) value r(n) = chi2inv(.95,2) / chi2inv(.95,n) so
// that r(n) * dist < chi2inv(2)  <->  dist < chi2inv(n)
inline double
mahalanobisMultiplicativeFactor(int dof)
{
    // return 1.618*std::pow(dof, -0.6288) - 0.05175;

    double num = 4.934 * dof + 19.59;
    double den = (dof + 10.61) * dof + 4.125;
    return num / den;
}

/*
 * thanks to tenfour at http://stackoverflow.com/questions/7571937
 */
template<template<typename, typename> class Container,
         typename Value,
         typename Allocator = std::allocator<Value>>
inline Container<Value, Allocator>
erase_indices(const Container<Value, Allocator>& data,
              std::vector<size_t>& indicesToDelete)
{
    if (indicesToDelete.empty())
        return data;

    Container<Value, Allocator> ret;
    ret.reserve(data.size() - indicesToDelete.size());

    std::sort(indicesToDelete.begin(), indicesToDelete.end());

    // new we can assume there is at least 1 element to delete. copy blocks at a
    // time.
    typename Container<Value, Allocator>::const_iterator itBlockBegin =
      data.begin();
    for (std::vector<size_t>::const_iterator it = indicesToDelete.begin();
         it != indicesToDelete.end();
         ++it) {
        typename Container<Value, Allocator>::const_iterator itBlockEnd =
          data.begin() + *it;
        if (itBlockBegin != itBlockEnd) {
            std::copy(itBlockBegin, itBlockEnd, std::back_inserter(ret));
        }
        itBlockBegin = itBlockEnd + 1;
    }

    // copy last block.
    if (itBlockBegin != data.end()) {
        std::copy(itBlockBegin, data.end(), std::back_inserter(ret));
    }

    return ret;
}

inline Eigen::MatrixXd
erase_column_indices(const Eigen::MatrixXd& data, std::vector<size_t>& indices);

inline Eigen::MatrixXd
erase_row_indices(const Eigen::MatrixXd& data, std::vector<size_t>& indices);

template<typename T>
struct RosParamType
{
    using type = T;
};

template<>
struct RosParamType<size_t>
{
    using type = int;
};

template<>
struct RosParamType<float>
{
    using type = double;
};

/*
 * For a vector of items {p1, p2, ..., pn}, produces a vector
 * of all pairwise combination of these items,
 * {(p1,p1), (p1, p2), ..., (p1, pn), (p2, p1), ..., (pn, pn)}
 */
template<typename T>
std::vector<std::pair<T, T>>
produceAllPairs(const std::vector<T>& items)
{
    std::vector<std::pair<T, T>> result;

    for (size_t i = 0; i < items.size(); ++i) {
        for (size_t j = i; j < items.size(); ++j) {
            result.push_back(std::make_pair(items[i], items[j]));
        }
    }

    return result;
}

// template<typename T>
// bool
// getRosParam(const ros::NodeHandle& nh, const std::string& name, T& out)
// {
//     typename RosParamType<T>::type val;
//     bool succeeded = nh.getParam(name, val);
//     out = val;

//     if (succeeded) {
//         return true;
//     } else {
//         ROS_ERROR_STREAM("Unable to read parameter " << name);
//         return false;
//     }
// }

// template<typename T>
// bool
// getRosParam(const ros::NodeHandle& nh,
//             const std::string& name,
//             T& out,
//             const T& default_val,
//             bool print_warning = true)
// {
//     typename RosParamType<T>::type val;
//     bool succeeded = nh.getParam(name, val);
//     out = val;

//     if (succeeded) {
//         return true;
//     } else {
//         if (print_warning)
//             ROS_WARN_STREAM("Unable to read parameter "
//                             << name << ", using default value.");
//         out = default_val;
//         return false;
//     }
// }

template<size_t Rows, size_t Cols>
void
boostArrayToEigen(const boost::array<double, Rows * Cols>& array,
                  Eigen::Matrix<double, Rows, Cols>& matrix)
{
    // assume that the boost array is row major, as ROS provides
    matrix =
      Eigen::Map<const Eigen::Matrix<double, Rows, Cols, Eigen::RowMajor>>(
        array.elems);
}

template<size_t Rows, size_t Cols>
void
eigenToBoostArray(const Eigen::Matrix<double, Rows, Cols>& matrix,
                  boost::array<double, Rows * Cols>& array)
{
    // copy in row major order
    for (size_t i = 0; i < Rows; i++) {
        for (size_t j = 0; j < Cols; j++) {
            array[i * Cols + j] = matrix(i, j);
        }
    }
}

template<size_t Rows, size_t Cols>
void
eigenToBoostArray(const Eigen::MatrixXd& matrix,
                  boost::array<double, Rows * Cols>& array)
{
    // copy in row major order
    for (size_t i = 0; i < Rows; i++) {
        for (size_t j = 0; j < Cols; j++) {
            array[i * Cols + j] = matrix(i, j);
        }
    }
}

// void
// FromROSMsg(const geometry_msgs::PoseWithCovariance& msg,
//            Eigen::Vector3d& position,
//            Eigen::Quaterniond& orientation,
//            Eigen::Matrix<double, 6, 6>& covariance);

// void FromROSMsg(const geometry_msgs::PoseWithCovariance& msg,
//                 gtsam::Pose3& pose,
//                 Eigen::Matrix<double, 6, 6>& covariance);

// void
// FromROSMsg(const sensor_msgs::Imu& msg,
//            Eigen::Vector3d& omega,
//            Eigen::Matrix3d& omega_cov,
//            Eigen::Vector3d& accel,
//            Eigen::Matrix3d& accel_cov);

// Eigen::MatrixXd removeRow(const Eigen::MatrixXd& A, int row);

// Eigen::MatrixXd removeCol(const Eigen::MatrixXd& A, int col);

/**
 * @brief Computes the relative pose difference of two poses
 * @details Given two poses xj = [pj qj] and qi = [pi qi], computes the relative
 * pose difference of the two, i.e. x_diff = xj (-) xi.
 */
// void relpose_difference(const Eigen::Vector3d& pj,
//                         const Eigen::Quaterniond& qj,
//                         const Eigen::Vector3d& pi,
//                         const Eigen::Quaterniond& qi,
//                         Eigen::Vector3d& p_diff,
//                         Eigen::Quaterniond& q_diff) {
//     p_diff = qi.inverse() * (pj - pi);
//     q_diff = qj.inverse() * qi;
// }
