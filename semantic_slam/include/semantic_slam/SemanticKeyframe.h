#pragma once

#include "semantic_slam/Common.h"

#include "semantic_slam/Symbol.h"

#include <eigen3/Eigen/Core>
#include <map>
// #include <ros/ros.h>
#include <vector>

#include <boost/enable_shared_from_this.hpp>

// using TimePoint = std::chrono::steady_clock::time_point;

class FactorGraph;
class EstimatedObject;
class CeresFactor;
class SE3Node;
class ImuBiasNode;

template<int Dim>
class VectorNode;

class SemanticKeyframe : public boost::enable_shared_from_this<SemanticKeyframe>
{
  public:
    using This = SemanticKeyframe;
    using Ptr = boost::shared_ptr<This>;

    SemanticKeyframe(Key key, TimePoint time, bool include_inertial = false);

    Key key() const { return key_; }
    Key bias_key() const { return Symbol('b', index()); }
    int index() const { return Symbol(key_).index(); }
    unsigned char chr() const { return Symbol(key_).chr(); }

    TimePoint time() const { return time_; }

    Pose3& pose() { return pose_; }
    const Pose3& pose() const { return pose_; }

    Eigen::MatrixXd& covariance() { return pose_covariance_; }
    const Eigen::MatrixXd& covariance() const { return pose_covariance_; }

    Pose3& odometry() { return odometry_; }
    const Pose3& odometry() const { return odometry_; }

    Eigen::MatrixXd& odometry_covariance() { return odometry_covariance_; }
    const Eigen::MatrixXd& odometry_covariance() const
    {
        return odometry_covariance_;
    }

    boost::shared_ptr<CeresFactor>& spine_factor() { return spine_factor_; }
    const boost::shared_ptr<CeresFactor>& spine_factor() const
    {
        return spine_factor_;
    }

    boost::shared_ptr<SE3Node>& graph_node() { return graph_node_; }
    const boost::shared_ptr<SE3Node>& graph_node() const { return graph_node_; }

    bool& loop_closing() { return loop_closing_; }
    const bool& loop_closing() const { return loop_closing_; }

    bool inGraph() const { return in_graph_; }

    void addToGraph(boost::shared_ptr<FactorGraph> graph);

    void updateConnections();
    void updateGeometricConnections();

    void addConnection(SemanticKeyframe::Ptr other, int weight);
    void addGeometricConnection(SemanticKeyframe::Ptr other, int weight);

    bool& measurements_processed() { return measurements_processed_; }
    const bool& measurements_processed() const
    {
        return measurements_processed_;
    }

    bool& covariance_computed_exactly() { return covariance_computed_exactly_; }

    const aligned_vector<ObjectMeasurement>& measurements() const
    {
        return measurements_;
    }

    aligned_vector<ObjectMeasurement>& measurements() { return measurements_; }

    const std::map<SemanticKeyframe::Ptr, int>& neighbors() const
    {
        return neighbors_;
    }

    std::vector<std::unordered_map<int, double>>& association_weights()
    {
        return association_weights_;
    }
    const std::vector<std::unordered_map<int, double>>& association_weights()
      const
    {
        return association_weights_;
    }

    const std::map<SemanticKeyframe::Ptr, int>& geometric_neighbors() const
    {
        return geometric_neighbors_;
    }

    // std::vector<ObjectMeasurement>& measurements() { return measurements_; }
    std::vector<boost::shared_ptr<EstimatedObject>>& visible_objects()
    {
        return visible_objects_;
    }

    std::vector<boost::shared_ptr<GeometricFeature>>&
    visible_geometric_features()
    {
        return visible_geometric_features_;
    }

    TimePoint image_time;

    // Only used/valid if odometry source is inertial
    boost::shared_ptr<VectorNode<3>>& velocity_node() { return velocity_node_; }
    const boost::shared_ptr<VectorNode<3>>& velocity_node() const
    {
        return velocity_node_;
    }

    boost::shared_ptr<ImuBiasNode>& bias_node() { return bias_node_; }
    const boost::shared_ptr<ImuBiasNode>& bias_node() const
    {
        return bias_node_;
    }

    Eigen::Vector3d& velocity() { return velocity_; }
    const Eigen::Vector3d& velocity() const { return velocity_; }

    Eigen::Matrix<double, 6, 1>& bias() { return bias_; }
    const Eigen::Matrix<double, 6, 1>& bias() const { return bias_; }

    Eigen::Matrix<double, 6, 6>& bias_covariance() { return bias_covariance_; }
    const Eigen::Matrix<double, 6, 6>& bias_covariance() const
    {
        return bias_covariance_;
    }

  private:
    Key key_;
    TimePoint time_;

    bool in_graph_;

    aligned_vector<ObjectMeasurement> measurements_;
    bool measurements_processed_;

    // association_weights_[i] contains the set of data association weights for
    // measurements_[i], i.e. association_weights_[i][j] is the weight of the
    // link between measurements_[i] and map object j. We're using an
    // unordered_map because some objects may be absent, so j not in
    // association_weights_[i] is equivalent to a weight of zero.
    std::vector<std::unordered_map<int, double>> association_weights_;

    Pose3 odometry_;
    Eigen::MatrixXd odometry_covariance_;
    Pose3 pose_;
    Eigen::MatrixXd pose_covariance_;

    bool covariance_computed_exactly_;

    boost::shared_ptr<SE3Node> graph_node_;

    boost::shared_ptr<CeresFactor> spine_factor_;

    // true if a loop closure was detected in this keyframe
    bool loop_closing_;

    bool include_inertial_;

    // aligned_vector<ObjectMeasurement> measurements_;
    std::vector<boost::shared_ptr<EstimatedObject>> visible_objects_;

    std::vector<GeometricFeature::Ptr> visible_geometric_features_;

    // Connections to keyframes that observe the same objects along with the
    // number of mutually observed objects can't use unordered_map without
    // custom hash function bc std::hash<boost::shared_ptr> is not defined
    std::map<SemanticKeyframe::Ptr, int> neighbors_;

    std::map<SemanticKeyframe::Ptr, int> geometric_neighbors_;

    // Only used/valid if odometry source is inertial
    Eigen::Vector3d velocity_;
    Eigen::Matrix<double, 6, 1> bias_;
    Eigen::Matrix<double, 6, 6> bias_covariance_;

    boost::shared_ptr<VectorNode<3>> velocity_node_;
    boost::shared_ptr<ImuBiasNode> bias_node_;

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};
