
#include "semantic_slam/ProbDataAssociator.h"
#include "semantic_slam/munkres.h"
#include "semantic_slam/permanent.h"

#include <chrono>


void removeRowCol(const Eigen::MatrixXd matrix, unsigned int rowToRemove, unsigned int colToRemove, Eigen::MatrixXd& result){
    unsigned int numRows = matrix.rows();
    unsigned int numCols = matrix.cols();

    result.block(0, 0, rowToRemove, colToRemove) = matrix.block(0, 0, rowToRemove, colToRemove);

    result.block(0, colToRemove, rowToRemove, numCols - colToRemove -1) = 
    matrix.block(0, colToRemove+1, rowToRemove, numCols - colToRemove -1);

    result.block(rowToRemove, 0, numRows - rowToRemove - 1, colToRemove) = 
    matrix.block(rowToRemove+1, 0, numRows - rowToRemove - 1, colToRemove);

    result.block(rowToRemove, colToRemove, numRows - rowToRemove - 1, numCols - colToRemove -1) = 
    matrix.block(rowToRemove+1, colToRemove+1, numRows - rowToRemove - 1, numCols - colToRemove -1);
}

void removeRows(const Eigen::MatrixXd matrix, std::vector<size_t> rowsToRemove, Eigen::MatrixXd& result)
{
    size_t numRows = matrix.rows();
    size_t numCols = matrix.cols();

    size_t idx = 0;
    for (size_t i = 0; i < numRows; ++i) {
        if (std::find(rowsToRemove.begin(), rowsToRemove.end(), i) != rowsToRemove.end())
            continue;
        
        result.row(idx) = matrix.row(i);
        idx++;

    }
}

// Eigen::MatrixXd
// ProbDataAssociator::computeConstraintWeights(const Eigen::MatrixXd& mahals)
// {
//     int m = mahals.rows();
//     int n = mahals.cols();

//     if (m == 0) {
//         return Eigen::MatrixXd::Zero(m, n + 1);
//     }

//     Eigen::MatrixXd weights = Eigen::MatrixXd::Zero(m, n + 1);

//     if (n == 0) {
//         weights.block(0, 0, m, 1) = Eigen::MatrixXd::Ones(m, 1);
//         return weights;
//     }
    

//     Eigen::MatrixXd probs = (-0.5 * mahals).array().exp();

//     // for (int i = 0; i < m; ++i){
//     //     for (int j = 0; j < n; ++j){
//     //         probs(i, j) = probs(i, j) / sqrt(pow(2*M_PI, dim) * covDets(i, j));
//     //     }
//     // }

//     // std::cout << "probs max and min: " << probs.maxCoeff() << ", " << probs.minCoeff() << std::endl;

//     Eigen::MatrixXd probs_row_sum = probs.rowwise().sum();
//     std::cout << "probs_row_sum: \n" << probs_row_sum << std::endl;
//     Eigen::MatrixXd extended_row_sum = Eigen::MatrixXd::Zero(m, n);
//     // extended_row_sum << probs_row_sum, probs_row_sum, probs_row_sum;

//     for (int j = 0; j < n; ++j){
//         extended_row_sum.block(0, j, m, 1) = probs_row_sum;
//     }
//     // std::cout << "extended_row_sum: \n" << extended_row_sum << std::endl;
//     probs = probs.array() / extended_row_sum.array();
    
//     // Compute the min of each row in mahals to check for new landmark
//     // assignment
//     Eigen::VectorXd msmt_mahal_mins;
//     if (n > 0) {
//         msmt_mahal_mins = mahals.rowwise().minCoeff();
//     }

//     if (m == 1) {
//         if (msmt_mahal_mins(0) > params_.mahal_thresh_init) {
//             weights(0, n) = 1;
//         }
//         else {
//             weights.block(0, 0, m, n) = probs;
//         }
//         return weights;
//     }
    
//     if (n == 1) {
//         for (int i = 0; i < weights.rows(); ++i) {
//             if (msmt_mahal_mins(i) > params_.mahal_thresh_init) {
//                 weights(i, n) = 1;
//             }
//             else {
//                 weights(i, 0) = 1;
//             }
//         }
//         return weights;
//     }

//     double overall_perma = permanentExact(probs);
//     // double overall_perma = permanentFastest(probs);
//     // double overall_perma = permanentApproximation(probs, 300);
//     Eigen::MatrixXd probs_sub = Eigen::MatrixXd::Zero(m-1, n-1);

//     for (int i = 0; i < weights.rows(); ++i) {

//         if (msmt_mahal_mins(i) > params_.mahal_thresh_init) {
//             weights(i, n) = 1;
//             continue;
//         }

//         for (int j = 0; j < weights.cols() - 1; ++j) {
//             // double prob_ij = probs(i, j);
            
//             removeRowCol(probs, i, j, probs_sub);
//             weights(i, j) = probs(i, j) * permanentExact(probs_sub) / overall_perma;
//             // weights(i, j) = probs(i, j) * permanentFastest(probs_sub) / overall_perma;
//             // weights(i, j) = probs(i, j) * permanentApproximation(probs_sub, 300) / overall_perma;
//         }
//     }

//     return weights;
// }


Eigen::MatrixXd
ProbDataAssociator::computeConstraintWeights(const Eigen::MatrixXd& mahals)
{
    int m = mahals.rows();
    int n = mahals.cols();

    if (m == 0) {
        return Eigen::MatrixXd::Zero(m, n + 1);
    }

    Eigen::MatrixXd weights = Eigen::MatrixXd::Zero(m, n + 1);

    if (n == 0) {
        weights.block(0, 0, m, 1) = Eigen::MatrixXd::Ones(m, 1);
        return weights;
    }
    
    Eigen::MatrixXd probs = (-0.5 * mahals).array().exp();

    // for (int i = 0; i < m; ++i){
    //     for (int j = 0; j < n; ++j){
    //         probs(i, j) = probs(i, j) / sqrt(pow(2*M_PI, dim) * covDets(i, j));
    //     }
    // }
    // std::cout << "probs max and min: " << probs.maxCoeff() << ", " << probs.minCoeff() << std::endl;

    Eigen::MatrixXd probs_row_sum = probs.rowwise().sum();
    // std::cout << "probs_row_sum: \n" << probs_row_sum << std::endl;
    Eigen::MatrixXd extended_row_sum = Eigen::MatrixXd::Zero(m, n);
    // extended_row_sum << probs_row_sum, probs_row_sum, probs_row_sum;

    for (int j = 0; j < n; ++j){
        extended_row_sum.block(0, j, m, 1) = probs_row_sum;
    }
    // std::cout << "extended_row_sum: \n" << extended_row_sum << std::endl;
    probs = probs.array() / extended_row_sum.array();
    
    // Compute the min of each row in mahals to check for new landmark
    // assignment
    Eigen::VectorXd msmt_mahal_mins;
    if (n > 0) {
        msmt_mahal_mins = mahals.rowwise().minCoeff();
    }

    if (m == 1) {
        if (msmt_mahal_mins(0) > params_.mahal_thresh_init) {
            weights(0, n) = 1;
        }
        else {
            weights.block(0, 0, m, n) = probs;
        }
        return weights;
    }
    
    if (n == 1) {
        for (int i = 0; i < weights.rows(); ++i) {
            if (msmt_mahal_mins(i) > params_.mahal_thresh_init) {
                weights(i, n) = 1;
            }
            else {
                weights(i, 0) = 1;
            }
        }
        return weights;
    }


    std::vector<size_t> rowsToRemove;
    for (size_t i = 0; i < weights.rows(); ++i) {
        if (msmt_mahal_mins(i) > params_.mahal_thresh_init) {
            weights(i, n) = 1;
            rowsToRemove.push_back(i);
            // continue;
        }

        // for (int j = 0; j < weights.cols() - 1; ++j) {
        //     removeRowCol(probs, i, j, probs_sub);
        //     weights(i, j) = probs(i, j) * permanentFastest(probs_sub) / overall_perma;
        // }
    }

    if (rowsToRemove.size() == weights.rows()) {
        return weights;    
    }

    Eigen::MatrixXd probs_left(probs.rows() - rowsToRemove.size(), probs.cols());

    removeRows(probs, rowsToRemove, probs_left);
    Eigen::MatrixXd probs_sub = Eigen::MatrixXd::Zero(probs_left.rows() - 1, probs_left.cols() - 1);

    // double overall_perma = permanentExact(probs_left);
    double overall_perma = permanentFastest(probs_left);
    // double overall_perma = permanentApproximation(probs, 300);
    // Eigen::MatrixXd probs_sub = Eigen::MatrixXd::Zero(m-1, n-1);

    size_t idx = 0;
    for (size_t i = 0; i < weights.rows(); ++i) {
        if (std::find(rowsToRemove.begin(), rowsToRemove.end(), i) != rowsToRemove.end())
            continue;
        
        for (int j = 0; j < weights.cols() - 1; ++j) {
            // auto tstart = std::chrono::high_resolution_clock::now();
            removeRowCol(probs_left, idx, j, probs_sub);
            // auto duration = std::chrono::high_resolution_clock::now() - tstart;
            // std::cout << "removeRowCol time taken at time: " << std::chrono::duration_cast<std::chrono::microseconds>(duration).count() / 1000. << " ms" << std::endl;
            // fflush(stdout);

            // weights(i, j) = probs_left(idx, j) * permanentFastest(probs_sub) / overall_perma;
            // tstart = std::chrono::high_resolution_clock::now();
            weights(i, j) = probs_left(idx, j) * permanentFastest(probs_sub) / overall_perma;
            // duration = std::chrono::high_resolution_clock::now() - tstart;
            // std::cout << "sub-permanent time taken at time: " << std::chrono::duration_cast<std::chrono::microseconds>(duration).count() / 1000. << " ms" << std::endl;
            // fflush(stdout);
        }
        idx += 1;
    }

    return weights;
}


// Eigen::MatrixXd
// ProbDataAssociator::computeWeightsWithPermanent(const Eigen::MatrixXd& mahals, const Eigen::MatrixXd& covDets, int dim)
// {
//     int m = mahals.rows();
//     int n = mahals.cols();

//     if (m == 0) {
//         return Eigen::MatrixXd::Zero(m, n + 1);
//     }

//     Eigen::MatrixXd weights = Eigen::MatrixXd::Zero(m, n + 1);

//     if (n == 0) {
//         weights.block(0, 0, m, 1) = Eigen::MatrixXd::Ones(m, 1);
//         return weights;
//     }
    
//     Eigen::MatrixXd probs = (-0.5 * mahals).array().exp();

//     // std::cout << "probs before divided by det:" << std::endl;
//     // std::cout << probs << std::endl;

//     for (int i = 0; i < m; ++i){
//         for (int j = 0; j < n; ++j){
//             probs(i, j) = probs(i, j) / sqrt(pow(2*M_PI, dim) * covDets(i, j));
//         }
//     }

//     // std::cout << "probs max and min: " << probs.maxCoeff() << ", " << probs.minCoeff() << std::endl;

//     // Eigen::MatrixXd probs_row_sum = probs.rowwise().sum();
//     // std::cout << "probs_row_sum: \n" << probs_row_sum << std::endl;
//     // Eigen::MatrixXd extended_row_sum = Eigen::MatrixXd::Zero(m, n);
//     // // extended_row_sum << probs_row_sum, probs_row_sum, probs_row_sum;

//     // for (int j = 0; j < n; ++j){
//     //     extended_row_sum.block(0, j, m, 1) = probs_row_sum;
//     // }
//     // std::cout << "extended_row_sum: \n" << extended_row_sum << std::endl;

//     // probs = probs.array() / extended_row_sum.array();
    
//     std::cout << "probs:" << std::endl;
//     std::cout << probs << std::endl;

//     // Compute the min of each row in mahals to check for new landmark
//     // assignment
//     Eigen::VectorXd msmt_mahal_mins;
//     if (n > 0) {
//         msmt_mahal_mins = mahals.rowwise().minCoeff();
//     }

//     if (m == 1) {
//         if (msmt_mahal_mins(0) > params_.mahal_thresh_init) {
//             weights(0, n) = 1;
//         }
//         else {
//             double overall_perma = permanentFastest(probs);
//             weights.block(0, 0, m, n) = probs * (1/overall_perma);
//             // weights.block(0, 0, m, n) = probs;
//         }
//         return weights;
//     }
    
//     if (n == 1) {
//         for (int i = 0; i < weights.rows(); ++i) {
//             if (msmt_mahal_mins(i) > params_.mahal_thresh_init) {
//                 weights(i, n) = 1;
//             }
//             else {
//                 weights(i, 0) = 1;
//             }
//         }
//         return weights;
//     }

//     double overall_perma = permanentExact(probs);
//     // double overall_perma = permanentFastest(probs);
//     // double overall_perma = permanentApproximation(probs, 300);
//     if (overall_perma < 0) {
//         std::cout << "negative overall permanent" << std::endl;
//     }

//     if (overall_perma == 0) {
//         std::cout << "zero overall permanent" << std::endl;
//         std::cout << "probs sum: " << probs.sum() << std::endl;
//         std::cout << "max prob: " << probs.maxCoeff() << " min prob: " << probs.minCoeff() << std::endl;
//         std::cout << "probs:" << std::endl;
//         std::cout << probs << std::endl;
//     }

//     Eigen::MatrixXd probs_sub = Eigen::MatrixXd::Zero(m-1, n-1);

//     for (int i = 0; i < weights.rows(); ++i) {
//         if (msmt_mahal_mins(i) > params_.mahal_thresh_init) {
//             weights(i, n) = 1;
//             continue;
//         }

//         for (int j = 0; j < weights.cols() - 1; ++j) {
//             // double prob_ij = probs(i, j);
//             removeRowCol(probs, i, j, probs_sub);
//             // weights(i, j) = probs(i, j) * permanentExact(probs_sub) / overall_perma;
//             if (probs_sub.minCoeff() < 0){
//                 std::cout << "negative probability" << std::endl;
//             }
//             if (permanentFastest(probs_sub) < 0) {
//                 std::cout << "negative permanent" << std::endl;
//             }
//             weights(i, j) = probs(i, j) * permanentFastest(probs_sub) / overall_perma;
//             // weights(i, j) = probs(i, j) * permanentApproximation(probs_sub, 300) / overall_perma;
//         }
//     }


//     // debug
//     // std::cout << "Munkres result is: " << std::endl << munkres_result <<
//     // std::endl;

//     // std::cout << "probs max and min: " << probs.maxCoeff() << ", " << probs.minCoeff() << std::endl;
//     // std::cout << "probs: \n" << probs << std::endl;

//     // std::cout << "mahals max and min: " << mahals.maxCoeff() << ", " << mahals.minCoeff() << std::endl;
//     // std::cout << "mahals: \n" << mahals << std::endl;
    
//     // std::cout << "weights max and min: " << weights.maxCoeff() << ", " << weights.minCoeff() << std::endl;
//     // std::cout << "weights: \n" << weights << std::endl;
//     return weights;
// }


Eigen::MatrixXd
ProbDataAssociator::computeWeightsWithPermanent(const Eigen::MatrixXd& mahals, const Eigen::MatrixXd& covDets, int dim, bool& valid)
{
    int m = mahals.rows();
    int n = mahals.cols();

    if (m == 0) {
        return Eigen::MatrixXd::Zero(m, n + 1);
    }

    Eigen::MatrixXd weights = Eigen::MatrixXd::Zero(m, n + 1);

    if (n == 0) {
        weights.block(0, 0, m, 1) = Eigen::MatrixXd::Ones(m, 1);
        return weights;
    }
    
    Eigen::MatrixXd probs = (-0.5 * mahals).array().exp();

    // std::cout << "probs before divided by det:" << std::endl;
    // std::cout << probs << std::endl;

    /********************************** consider covariance determinant *************************************/
    for (int i = 0; i < m; ++i){
        for (int j = 0; j < n; ++j){
            probs(i, j) = probs(i, j) / sqrt(pow(2*M_PI, dim) * covDets(i, j));
        }
    }

    // std::cout << "probs max and min: " << probs.maxCoeff() << ", " << probs.minCoeff() << std::endl;

    /*********************************** normalize each row ************************************************/
    // Eigen::MatrixXd probs_row_sum = probs.rowwise().sum();
    // // std::cout << "probs_row_sum: \n" << probs_row_sum << std::endl;
    // Eigen::MatrixXd extended_row_sum = Eigen::MatrixXd::Zero(m, n);

    // for (int j = 0; j < n; ++j){
    //     extended_row_sum.block(0, j, m, 1) = probs_row_sum;
    // }
    // std::cout << "extended_row_sum: \n" << extended_row_sum << std::endl;

    // probs = probs.array() / extended_row_sum.array();

    /**************************** normalize by dividing by the max element *********************************/
    probs = probs.array() / probs.maxCoeff();

    // Compute the min of each row in mahals to check for new landmark
    // assignment
    Eigen::VectorXd msmt_mahal_mins;
    if (n > 0) {
        msmt_mahal_mins = mahals.rowwise().minCoeff();
    }

    if (m == 1) {
        if (msmt_mahal_mins(0) > params_.mahal_thresh_init) {
            weights(0, n) = 1;
        }
        else {
            if (msmt_mahal_mins(0) <= params_.mahal_thresh_assign) {
                double overall_perma = permanentFastest(probs);
                weights.block(0, 0, m, n) = probs * (1/overall_perma);
            }
            // weights.block(0, 0, m, n) = probs;
        }
        return weights;
    }
    
    if (n == 1) {
        for (int i = 0; i < weights.rows(); ++i) {
            if (msmt_mahal_mins(i) > params_.mahal_thresh_init) {
                weights(i, n) = 1;
            }
            else {
                weights(i, 0) = 1;
            }
        }
        return weights;
    }
    
    std::vector<size_t> rowsToRemove;
    for (int i = 0; i < weights.rows(); ++i) {
        if (msmt_mahal_mins(i) > params_.mahal_thresh_init) {
            weights(i, n) = 1;
            rowsToRemove.push_back(i);
            // continue;
        }

        // for (int j = 0; j < weights.cols() - 1; ++j) {
        //     // double prob_ij = probs(i, j);
        //     removeRowCol(probs, i, j, probs_sub);
        //     // weights(i, j) = probs(i, j) * permanentExact(probs_sub) / overall_perma;
        //     if (probs_sub.minCoeff() < 0){
        //         std::cout << "negative probability" << std::endl;
        //     }
        //     if (permanentFastest(probs_sub) < 0) {
        //         std::cout << "negative permanent" << std::endl;
        //     }
        //     weights(i, j) = probs(i, j) * permanentFastest(probs_sub) / overall_perma;
        //     // weights(i, j) = probs(i, j) * permanentApproximation(probs_sub, 300) / overall_perma;
        // }
    }

    if (rowsToRemove.size() == weights.rows()) {
        return weights;    
    }

    /********************************** DEBUG: print probabilities *************************************/
    // std::cout << "probs:" << std::endl;
    // std::cout << probs << std::endl;
    // std::cout << "rows to remove:" << std::endl;
    // for (size_t i = 0; i < rowsToRemove.size(); ++i) {
    //     std::cout << rowsToRemove[i] << " ";
    // }
    // std::cout << std::endl;

    Eigen::MatrixXd probs_left(probs.rows() - rowsToRemove.size(), probs.cols());
    removeRows(probs, rowsToRemove, probs_left);
    Eigen::MatrixXd probs_sub = Eigen::MatrixXd::Zero(probs_left.rows() - 1, probs_left.cols() - 1);
    
    if (std::max(probs_left.rows(), probs_left.cols()) > 32) {
        std::cout << "large probs matrix: (" << probs_left.rows() << ", " << probs_left.cols() << ")" << std::endl;
        if (std::min(probs_left.rows(), probs_left.cols()) > 3) {
            std::cout << "very large probs matrix: (" << probs_left.rows() << ", " << probs_left.cols() << ")" << std::endl;
        }
    }

    // normalize each row
    // int m_left = probs_left.rows();
    // int n_left = probs_left.cols();

    // Eigen::MatrixXd probs_rowsum = probs_left.rowwise().sum();
    // Eigen::MatrixXd probs_rowsum_ext = Eigen::MatrixXd::Zero(m_left, n_left);
    // for (int i = 0; i < probs_rowsum_ext.cols(); ++i) {
    //     probs_rowsum_ext.block(0, i, m_left, 1) = probs_rowsum;
    // }
    // std::cout << "probs left before normalization: \n" << probs_left << std::endl;
    // probs_left = probs_left.array() / probs_rowsum_ext.array();
    // std::cout << "probs left after normalization: \n" << probs_left << std::endl;

    // std::cout << "probs left before normalization: \n" << probs_left << std::endl;
    
    // std::cout << "probs left after normalization: \n" << probs_left << std::endl;

    // double overall_perma;
    // if (std::max(probs_left.rows(), probs_left.cols()) <= 30)
    //     overall_perma = permanentExact(probs_left);
    // else
    //     overall_perma = permanentFastest(probs_left);
        // overall_perma = permanentApproximation(probs_left, 300);

    double overall_perma = permanentFastest(probs_left);
    // std::cout << "overall permanent: " << overall_perma << std::endl;

    if (overall_perma <= 0) {
        valid = false;
        return weights;
    }

    if (overall_perma < 0) {
        std::cout << "negative overall permanent" << std::endl;
        // std::cout << "mahals:" << std::endl;
        // std::cout << mahals << std::endl;
        // std::cout << "cov det:" << std::endl;
        // std::cout << covDets << std::endl;
        // std::cout << "probs:" << std::endl;
        // std::cout << probs_left << std::endl;
    }

    if (overall_perma == 0) {
        std::cout << "zero overall permanent" << std::endl;
        // std::cout << "probs sum: " << probs.sum() << std::endl;
        // std::cout << "max prob: " << probs.maxCoeff() << " min prob: " << probs.minCoeff() << std::endl;
        // std::cout << "probs:" << std::endl;
        // std::cout << probs_left << std::endl;
    }

    size_t idx = 0;
    for (size_t i = 0; i < weights.rows(); ++i) {
        if (std::find(rowsToRemove.begin(), rowsToRemove.end(), i) != rowsToRemove.end())
            continue;
        
        if (probs_left.rows() == 1) {
            for (int j = 0; j < weights.cols() - 1; ++j) {
                // std::cout << "prob(" << idx << ", " << j << "): " << probs_left(idx, j) << std::endl;

                if (mahals(i, j) < params_.mahal_thresh_assign) {
                    weights(i, j) = probs_left(idx, j) / overall_perma;
                }    
            }

            break;    
        }
        
        for (int j = 0; j < weights.cols() - 1; ++j) {
            removeRowCol(probs_left, idx, j, probs_sub);
            // std::cout << "removed row: " << idx << " and col: " << j << std::endl;
            // std::cout << "probs_sub:\n" << probs_sub << std::endl;
            // std::cout << "prob at (" << idx << ", " << j << ") is " << probs_left(idx, j) << std::endl;
            double sub_perma = permanentFastest(probs_sub);
            // double sub_perma = permanentExactPlain(probs_sub);
            
            // double sub_perma_exact = permanentExact(probs_sub);
            // if (abs(sub_perma - sub_perma_exact) / abs(sub_perma_exact) >= 0.01) std::cout << "unequal permanent!\n";
            
            // std::cout << "sub permanent: " << sub_perma << std::endl;
            // std::cout << "sub permanent_exact: " << sub_perma << std::endl;
            // std::cout << "prob(" << idx << ", " << j << "): " << probs_left(idx, j) << std::endl;

            if (mahals(i, j) < params_.mahal_thresh_assign) {
                weights(i, j) = probs_left(idx, j) * sub_perma / overall_perma;
            }
            
            // weights(i, j) = probs_left(idx, j) * permanentExact(probs_sub) / overall_perma;
        }
        idx += 1;
    }
    
    if (probs_left.rows() > 1) {
    
    }
    else {
        
    }
    
    return weights;
}

