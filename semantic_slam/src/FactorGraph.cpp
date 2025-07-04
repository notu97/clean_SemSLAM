
#include "semantic_slam/FactorGraph.h"

// #include <rosfmt/rosfmt.h>

#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>

FactorGraph::FactorGraph()
  : modified_(false)
{
    ceres::Problem::Options problem_options;
    problem_options.cost_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
    problem_options.enable_fast_removal = true;
    problem_ = boost::make_shared<ceres::Problem>(problem_options);

    // set covariance options if needed...
    // covariance_options_....?
    covariance_options_.apply_loss_function = true;
    covariance_options_.num_threads = 4;
    // covariance_options_.algorithm_type = ceres::SPARSE_CHOLESKY;
    covariance_ = boost::make_shared<ceres::Covariance>(covariance_options_);
}

void
FactorGraph::setSolverOptions(ceres::Solver::Options solver_options)
{
    solver_options_ = solver_options;
}

void
FactorGraph::setNumThreads(int n_threads)
{
    solver_options_.num_threads = n_threads;
    covariance_options_.num_threads = n_threads;
}

bool
FactorGraph::setNodeConstant(Key key)
{
    const auto& node = getNode(key);
    return setNodeConstant(node);
}

bool
FactorGraph::setNodeConstant(CeresNodePtr node)
{
    for (auto& block : node->parameter_blocks()) {
        problem_->SetParameterBlockConstant(block);
    }
    return true;
}

bool
FactorGraph::setNodesConstant(const std::vector<CeresNodePtr>& nodes)
{
    for (const auto& node : nodes) {
        setNodeConstant(node);
    }

    return true;
}

bool
FactorGraph::setNodeVariable(Key key)
{
    const auto& node = getNode(key);
    return setNodeVariable(node);
}

bool
FactorGraph::setNodeVariable(CeresNodePtr node)
{
    for (auto& block : node->parameter_blocks()) {
        problem_->SetParameterBlockVariable(block);
    }
    return true;
}

bool
FactorGraph::setNodesVariable(const std::vector<CeresNodePtr>& nodes)
{
    for (const auto& node : nodes) {
        setNodeVariable(node);
    }
    return true;
}

bool
FactorGraph::isNodeConstant(CeresNodePtr node) const
{
    // assume that the user is not interfacing with the ceres::Problem
    // directly... i.e. assume that one of the node's parameter blocks is
    // constant iff they all are.
    return problem_->IsParameterBlockConstant(node->parameter_blocks()[0]);
}

bool
FactorGraph::containsNode(CeresNodePtr node)
{
    return containsNode(node->key());
}

bool
FactorGraph::containsNode(Key key)
{
    return nodes_.find(key) != nodes_.end();
}

boost::shared_ptr<FactorGraph>
FactorGraph::clone() const
{
    auto new_graph = util::allocate_aligned<FactorGraph>();

    std::lock_guard<std::mutex> lock(mutex_);

    // Create the set of new nodes over which we'll be operating
    std::unordered_map<Key, CeresNodePtr> new_nodes;
    for (const auto& node : nodes_) {
        auto new_node = node.second->clone();

        new_nodes.emplace(node.first, new_node);
        new_graph->addNode(new_node);

        if (isNodeConstant(node.second)) {
            new_graph->setNodeConstant(new_node);
        }
    }

    // Need to iterate over each factor, clone it, and set it to operate on
    // the new nodes
    for (const auto& factor : factors_) {
        auto new_fac = factor->clone();

        // new_fac contains all the correct measurement info etc but is
        // currently set with all NULL nodes. collect the set of nodes based on
        // keys from the old factor
        std::vector<CeresNodePtr> new_factor_nodes;
        for (const auto& old_node : factor->nodes()) {
            new_factor_nodes.push_back(new_nodes[old_node->key()]);
        }

        new_fac->setNodes(new_factor_nodes);
        new_graph->addFactor(new_fac);
    }

    new_graph->setSolverOptions(solver_options_);

    // copying elimination ordering is not yet supported
    new_graph->solver_options().linear_solver_ordering = nullptr;

    return new_graph;
}

bool
FactorGraph::solve(bool verbose,
                   boost::optional<ceres::Solver::Summary&> summary_in)
{
    ceres::Solver::Summary summary;

    solver_options_.minimizer_progress_to_stdout = verbose;
    // std::cout << 1 << std::endl;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        // std::cout << 2 << std::endl;
        ceres::Solve(solver_options_, problem_.get(), &summary);
        modified_ = false;
    }
    
    if (verbose)
        std::cout << summary.FullReport() << std::endl;

    if (verbose) {
        std::cout << "Linear solver ordering sizes: \n";
        for (auto& sz : summary.linear_solver_ordering_used) {
            std::cout << sz << " ";
        }
        std::cout << std::endl;

        std::cout << "Schur structure detected: "
                  << summary.schur_structure_given << std::endl;
        std::cout << "Schur structure used: " << summary.schur_structure_used
                  << std::endl;
    }
    // std::cout << 3 << std::endl;
    if (summary_in)
        *summary_in = summary;
    // std::cout << 4 << std::endl;
    return summary.IsSolutionUsable();
    // return summary.termination_type == ceres::CONVERGENCE;
}

void
FactorGraph::addNode(CeresNodePtr node)
{
    if (!node)
        return;
    if (nodes_.find(node->key()) != nodes_.end()) {
        throw std::runtime_error(std::string("Tried to add already existing node with symbol ") +
          DefaultKeyFormatter(node->key()));
    }

    std::lock_guard<std::mutex> lock(mutex_);
    nodes_[node->key()] = node;
    node->addToProblem(problem_);
    modified_ = true;
}

void
FactorGraph::addNodes(const std::vector<CeresNodePtr>& nodes)
{
    for (auto& node : nodes) {
        addNode(node);
    }
}

void
FactorGraph::addFactorInternal(CeresFactorPtr factor)
{
    if (!factor)
        return;
    factors_.push_back(factor);
    factor->addToProblem(problem_);
}

void
FactorGraph::addFactor(CeresFactorPtr factor)
{
    std::lock_guard<std::mutex> lock(mutex_);
    addFactorInternal(factor);
    modified_ = true;
}

void
FactorGraph::addFactors(std::vector<CeresFactorPtr> factors)
{
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& factor : factors) {
        addFactorInternal(factor);
    }
    modified_ = true;
}

void
FactorGraph::removeNode(CeresNodePtr node)
{
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = nodes_.find(node->key());
    if (it != nodes_.end()) {
        nodes_.erase(it);
    }

    modified_ = true;

    // bool found = false;
    // for (auto it = nodes_.begin(); it != nodes_.end(); ++it) {
    //     if (node->key() == it->second->key()) {
    //         nodes_.erase(it);
    //         found = true;
    //         break;
    //     }
    // }

    // node->removeFromProblem(problem_);

    // modified_ = true;
}

void
FactorGraph::removeFactor(CeresFactorPtr factor)
{
    std::lock_guard<std::mutex> lock(mutex_);

    bool found = false;
    for (auto it = factors_.begin(); it != factors_.end(); ++it) {
        if (it->get() == factor.get()) {
            factors_.erase(it);
            found = true;
            break;
        }
    }

    if (found) {
        factor->removeFromProblem(problem_);
        modified_ = true;
    }
}

std::vector<Key>
FactorGraph::keys()
{
    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<Key> result;
    result.reserve(nodes_.size());
    for (auto& node : nodes_) {
        result.push_back(node.first);
    }
    return result;
}

bool
FactorGraph::containsFactor(CeresFactorPtr factor)
{
    std::lock_guard<std::mutex> lock(mutex_);

    auto fac_it = std::find(factors_.begin(), factors_.end(), factor);

    if (fac_it != factors_.end()) {
        return true;
    } else {
        return false;
    }
}

bool
FactorGraph::computeMarginalCovariance(const std::vector<Key>& keys)
{
    // Retrieve nodes and call other covariance computation function
    std::vector<CeresNodePtr> nodes;
    for (auto& key : keys) {
        auto node = getNode(key);
        if (node)
            nodes.push_back(node);
    }
    return computeMarginalCovariance(nodes);
}

bool
FactorGraph::computeMarginalCovariance(const std::vector<CeresNodePtr>& nodes)
{
    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<const double*> blocks;
    for (const CeresNodePtr& node : nodes) {
        blocks.insert(blocks.end(),
                      node->parameter_blocks().begin(),
                      node->parameter_blocks().end());
    }

    // std::vector<std::pair<const double*, const double*>> block_pairs =
    //   produceAllPairs(blocks);

    return covariance_->Compute(blocks, problem_.get());

    // std::vector<std::pair<const double*, const double*>> cov_blocks;

    // for (size_t node_i = 0; node_i < nodes.size(); ++node_i) {
    //     for (size_t block_i = 0; block_i <
    //     nodes[node_i]->parameter_blocks().size(); ++block_i) {

    //         for (size_t node_j = node_i; node_j < nodes.size(); ++node_j) {
    //             // If node_j == node_i, want block_j index to start at
    //             block_i.
    //             // Else, want it to start at 0.
    //             size_t block_j = node_j == node_i ? block_i : 0;
    //             for (; block_j < nodes[node_j]->parameter_blocks().size();
    //             ++block_j) {
    //                 cov_blocks.push_back(std::make_pair(nodes[node_i]->parameter_blocks()[block_i],
    //                                                     nodes[node_j]->parameter_blocks()[block_j]));
    //             }
    //         }
    //     }
    // // }

    // return covariance_->Compute(cov_blocks, problem_.get());
}

Eigen::MatrixXd
FactorGraph::getMarginalCovariance(const Key& key) const
{
    auto node = getNode(key);
    if (!node)
        throw std::runtime_error("Error: tried to get the covariance of a node "
                                 "not in the FactorGraph");
    return getMarginalCovariance({ node });
}

Eigen::MatrixXd
FactorGraph::getMarginalCovariance(const Key& key1, const Key& key2) const
{
    auto node1 = getNode(key1);
    auto node2 = getNode(key2);
    if (!node1 || !node2) {
        throw std::runtime_error("Error: tried to get the covariance of a node "
                                 "not in the FactorGraph");
    }

    return getMarginalCovariance({ node1, node2 });
}

Eigen::MatrixXd
FactorGraph::getMarginalCovariance(const std::vector<CeresNodePtr>& nodes) const
{
    // Collect pointers to parameter blocks and their sizes
    // TODO need to streamline / standardize how data is stored in these
    // CeresNode objects...
    std::vector<double*> parameter_blocks;
    std::vector<size_t> parameter_block_sizes;
    size_t full_dim = 0;
    size_t max_block_size = 0;

    for (auto& node : nodes) {
        for (size_t i = 0; i < node->parameter_blocks().size(); ++i) {
            parameter_blocks.push_back(node->parameter_blocks()[i]);
            parameter_block_sizes.push_back(
              node->parameter_block_local_sizes()[i]);

            full_dim += node->parameter_block_local_sizes()[i];
            max_block_size =
              std::max(max_block_size, node->parameter_block_local_sizes()[i]);
        }
    }

    using RowMajorMatrixXd =
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    Eigen::MatrixXd C(full_dim, full_dim);

    // data buffer for ceres to write each block into...
    // std::vector<> memory is guaranteed to be contiguous, use it for RAII
    // purposes
    std::vector<double> data_buffer_vec(max_block_size * max_block_size, 0.0);
    double* buf = &data_buffer_vec[0];

    // Begin filling in the covariance matrix
    size_t index_i = 0;
    size_t index_j = 0;
    for (size_t i = 0; i < parameter_blocks.size(); ++i) {
        for (size_t j = i; j < parameter_blocks.size(); ++j) {
            covariance_->GetCovarianceBlockInTangentSpace(
              parameter_blocks[i], parameter_blocks[j], buf);

            C.block(index_i,
                    index_j,
                    parameter_block_sizes[i],
                    parameter_block_sizes[j]) =
              Eigen::Map<RowMajorMatrixXd>(
                buf, parameter_block_sizes[i], parameter_block_sizes[j]);

            index_j += parameter_block_sizes[j];
        }

        index_i += parameter_block_sizes[i];
        index_j = index_i;
    }

    return C.selfadjointView<Eigen::Upper>();
}

void
FactorGraph::addIterationCallback(IterationCallbackType callback)
{
    auto callback_wrapper =
      boost::make_shared<IterationCallbackWrapper>(callback);

    iteration_callbacks_.push_back(callback_wrapper);
    solver_options_.callbacks.push_back(callback_wrapper.get());
}

// CeresNodePtr
// FactorGraph::findLastNodeBeforeTime(unsigned char symbol_chr, ros::Time time)
// {
//     if (nodes_.size() == 0)
//         return nullptr;

//     ros::Time last_time(0);
//     CeresNodePtr node = nullptr;

//     for (auto& key_node : nodes_) {
//         if (key_node.second->chr() != symbol_chr)
//             continue;

//         if (!key_node.second->time())
//             continue;

//         if (key_node.second->time() > last_time &&
//             key_node.second->time() <= time) {
//             last_time = *key_node.second->time();
//             node = key_node.second;
//         }
//     }

//     return node;
// }

// CeresNodePtr
// FactorGraph::findFirstNodeAfterTime(unsigned char symbol_chr, ros::Time time)
// {
//     if (nodes_.size() == 0)
//         return nullptr;

//     ros::Time first_time = ros::TIME_MAX;
//     CeresNodePtr node = nullptr;

//     for (auto& key_node : nodes_) {
//         if (key_node.second->chr() != symbol_chr)
//             continue;

//         if (!key_node.second->time())
//             continue;

//         if (key_node.second->time() <= first_time &&
//             key_node.second->time() >= time) {
//             first_time = *key_node.second->time();
//             node = key_node.second;
//         }
//     }

//     return node;
// }

// CeresNodePtr
// FactorGraph::findNearestNode(unsigned char symbol_chr, ros::Time time)
// {
//     if (nodes_.size() == 0)
//         return nullptr;

//     ros::Duration shortest_duration = ros::DURATION_MAX;
//     CeresNodePtr node = nullptr;

//     for (auto& key_node : nodes_) {
//         if (key_node.second->chr() != symbol_chr)
//             continue;

//         if (!key_node.second->time())
//             continue;

//         if (abs_duration(time - *key_node.second->time()) <=
//             shortest_duration) {
//             shortest_duration = abs_duration(time - *key_node.second->time());
//             node = key_node.second;
//         }
//     }

//     return node;
// }

boost::shared_ptr<gtsam::NonlinearFactorGraph>
FactorGraph::getGtsamGraph() const
{
    auto graph = util::allocate_aligned<gtsam::NonlinearFactorGraph>();

    for (auto factor : factors_) {
        // if (factor->active()) {
        //     bool good = true;
        //     for (auto& node : factor->nodes()) {
        //         if (!node->active()) {
        //             good = false;
        //             break;
        //         }
        //     }

        //     if (good) graph->push_back(factor->getGtsamFactor());
        // }

        // if (factor->active()) graph->push_back(factor->getGtsamFactor());
        if (factor->active())
            factor->addToGtsamGraph(graph);
    }

    return graph;
}

boost::shared_ptr<gtsam::Values>
FactorGraph::getGtsamValues() const
{
    auto values = util::allocate_aligned<gtsam::Values>();

    for (auto node : nodes_) {
        if (node.second->active())
            values->insert(node.first, *node.second->getGtsamValue());
    }

    return values;
}
