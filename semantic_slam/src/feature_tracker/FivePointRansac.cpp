#include <boost/make_shared.hpp>
#include <chrono>
#include <opencv2/imgproc/imgproc.hpp>

#include "semantic_slam/feature_tracker/FivePointRansac.h"

FivePointRansac::FivePointRansac(int n_hypotheses, double sqrt_samp_thresh)
  : n_hypotheses_(n_hypotheses)
  , calibrated_(false)
  , samp_thresh_(sqrt_samp_thresh * sqrt_samp_thresh)
{

    hypotheses_.reserve(n_hypotheses);

    for (int i = 0; i < n_hypotheses; i++) {
        hypotheses_.push_back(boost::make_shared<Hypothesis>());
    }
}

size_t
FivePointRansac::computeInliers(const std::vector<cv::Point2f>& points_A,
                                const std::vector<cv::Point2f>& points_B,
                                Eigen::Matrix<bool, 1, Eigen::Dynamic>& inliers)
{
    size_t n_points = points_A.size();

    if (n_points < 8) {
        // impossible to compute any estimate with less than 5 correspondences.
        // with 5 points exactly it's impossible to do outlier rejection too (5
        // points will of course fit the model they define) so reject anything
        // less than 6-8 or so
        inliers = Eigen::Matrix<bool, 1, Eigen::Dynamic>::Zero(1, n_points);
        return 0;
    }

    // normalize & undistort points

    std::vector<cv::Point2f> pointsA_norm, pointsB_norm;

    cv::undistortPoints(points_A, pointsA_norm, camera_matrix_, dist_coeffs_);
    cv::undistortPoints(points_B, pointsB_norm, camera_matrix_, dist_coeffs_);

    // convert vector<point> to EIgen

    Eigen::MatrixXd eigenA(3, n_points);
    Eigen::MatrixXd eigenB(3, n_points);

    for (size_t i = 0; i < n_points; ++i) {
        eigenA.col(i) << pointsA_norm[i].x, pointsA_norm[i].y, 1;
        eigenB.col(i) << pointsB_norm[i].x, pointsB_norm[i].y, 1;
    }

    return computeInliersNormalized(eigenA, eigenB, inliers);
}

size_t
FivePointRansac::computeInliersNormalized(
  const Eigen::MatrixXd& points_A,
  const Eigen::MatrixXd& points_B,
  Eigen::Matrix<bool, 1, Eigen::Dynamic>& inliers)
{

    int n_points = points_A.cols();

    int winner_1 = 0; // TODO ADD AN EXCEPTION FOR NO INLIERS
    int winner_2 = 0;

    int winner = -1;
    // int i = 0;

    // double sqrt_samp_threshold = 6.25e-4;
    // double sqrt_samp_threshold = 5e-4;
    // double samp_threshold = 1*pow(sqrt_samp_threshold,2);

    for (size_t i = 0; i < n_hypotheses_; ++i) {
        hypotheses_[i]->inliers.setZero();

        selectRandomSet(n_points, i);

        solveFivePoint(points_A, points_B, i);

        // std::cout << "Hypothesis..." << i << "total solutions ..." <<
        // hypothesis_storage_.at(i)->totalSolutions << "\n";

        // Iterate over found solutions, compute errors and select inliers

        for (int j = 0; j < hypotheses_[i]->totalSolutions; j++) {

            for (int k = 0; k < n_points; ++k) {
                double err = computeSampsonError(
                  points_A.col(k),
                  points_B.col(k),
                  hypotheses_[i]->solutions.block<3, 3>(0, 3 * j));

                if (err < samp_thresh_) {
                    (hypotheses_[i]->inliers)(j, 0)++;
                }
            }

            if (hypotheses_[i]->inliers(j, 0) > winner) {
                winner_1 = i;
                winner_2 = j;
                winner = hypotheses_[i]->inliers(j, 0);
            }
        }
    }

    if (winner == -1) {
        // no valid solutions found for any hypothesis
        inliers = Eigen::Matrix<bool, 1, Eigen::Dynamic>::Zero(1, n_points);
        return 0;
    }

    inliers = Eigen::Matrix<bool, 1, Eigen::Dynamic>::Ones(1, n_points);

    for (int k = 0; k < n_points; ++k) {
        double err = computeSampsonError(
          points_A.col(k),
          points_B.col(k),
          hypotheses_[winner_1]->solutions.block<3, 3>(0, 3 * winner_2));

        if (pow(err, 1) > samp_thresh_) {
            inliers(0, k) = 0;
        }
    }

    return winner;
}

void
FivePointRansac::setCameraCalibration(double fx,
                                      double fy,
                                      double s,
                                      double u0,
                                      double v0,
                                      double k1,
                                      double k2,
                                      double p1,
                                      double p2)
{
    camera_matrix_ = (cv::Mat_<double>(3, 3) << fx, 0, u0, 0, fy, v0, 0, 0, 1);

    dist_coeffs_ = (cv::Mat_<double>(1, 4) << k1, k2, p1, p2);

    calibrated_ = true;
}

double
FivePointRansac::computeSampsonError(const Eigen::Vector3d& p1,
                                     const Eigen::Vector3d& p2,
                                     const Eigen::Matrix3d& essentialMatrix)
{
    Eigen::Vector3d Fx1;
    Eigen::Vector3d Fx2;
    Fx1 = (essentialMatrix)*p1;
    Fx2 = essentialMatrix.transpose() * p2;
    return (double)(pow(p2.transpose() * essentialMatrix * p1, 2)) /
           ((Fx1(0) * Fx1(0)) + (Fx1(1) * Fx1(1)) + (Fx2(1) * Fx2(1)) +
            (Fx2(0) * Fx2(0)));
}

void
FivePointRansac::solveFivePoint(const Eigen::MatrixXd& points_A,
                                const Eigen::MatrixXd& points_B,
                                int selection)
{
    fivePoint(points_A,
              points_B,
              hypotheses_.at(selection)->set,
              &(hypotheses_.at(selection)->solutions),
              &(hypotheses_.at(selection)->totalSolutions));
}

void
FivePointRansac::selectRandomSet(int num_points, int hyp_index)
{
    for (int i = 0; i < 5; ++i) {
        bool is_unique;
        do {
            hypotheses_[hyp_index]->set(i) = rand() % num_points;
            is_unique = true;
            for (int j = 0; j < i; ++j) {
                if (hypotheses_[hyp_index]->set(j) ==
                    hypotheses_[hyp_index]->set(i)) {
                    is_unique = false;
                    break;
                }
            }
        } while (!is_unique);
    }
}
