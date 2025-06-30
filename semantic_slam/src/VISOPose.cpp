#include <deque>
#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "libviso2/viso_math.h"
#include "viso_pose/VISOPose.h"

VISOPose::VISOPose(double px_sigma)
  : px_sigma_(px_sigma)
  , got_calibration_(false)
  , pose_(Matrix::eye(4))
  , full_pose_(Matrix::eye(4))
  , initialized_(false)
  , out_file_("/dataset/viso_intermediate_poses.txt")
{

    I_T_C_ = Eigen::Matrix4d::Identity();
    C_T_I_ = Eigen::Matrix4d::Identity();

    // caminfo0_sub_ =
    //   boost::make_shared<message_filters::Subscriber<sensor_msgs::CameraInfo>>(
    //     pnh_, "/cam0/camera_info", 500);
    // caminfo1_sub_ =
    //   boost::make_shared<message_filters::Subscriber<sensor_msgs::CameraInfo>>(
    //     pnh_, "/cam1/camera_info", 500);
    // caminfo_sync_ = boost::make_shared<
    //   message_filters::TimeSynchronizer<sensor_msgs::CameraInfo,
    //                                     sensor_msgs::CameraInfo>>(
    //   *caminfo0_sub_, *caminfo1_sub_, 500);
    // caminfo_sync_->registerCallback(
    //   boost::bind(&VISOPose::caminfo_callback, this, _1, _2));
}

// void
// VISOPose::img0_callback(const sensor_msgs::ImageConstPtr& msg)
// {
//     if (msg->header.seq != last_img0_seq_ + 1) {
//         ROS_ERROR("[VISOPose] Error: dropped left camera message");
//     }
//     img0_queue_.push_back(msg);
//     last_img0_seq_ = msg->header.seq;
//     tryProcessNextImages();
// }

// void
// VISOPose::img1_callback(const sensor_msgs::ImageConstPtr& msg)
// {
//     if (msg->header.seq != last_img1_seq_ + 1) {
//         ROS_ERROR("[VISOPose] Error: dropped right camera message");
//     }
//     img1_queue_.push_back(msg);
//     last_img1_seq_ = msg->header.seq;
//     tryProcessNextImages();
// }

void
VISOPose::setVisoParams(CameraInfo& cam0_info, CameraInfo& cam1_info)
{
    if (got_calibration_)
        return;

    // std::cout<< "inside setVisoParams" << std::endl;

    viso_params_.calib.f = cam0_info.P[0];
    viso_params_.calib.cu = cam0_info.P[2];
    viso_params_.calib.cv = cam0_info.P[6];
    viso_params_.base =
      std::fabs((cam1_info.P[3] - cam0_info.P[3]) / viso_params_.calib.f);
    viso_params_.px_noise = px_sigma_;
    got_calibration_ = true;
    // std::cout << "caminfo callback" << std::endl;
    viso_ = boost::make_shared<VisualOdometryStereo>(viso_params_);

    if (!viso_) {
        std::cerr << "viso_ is not initialized!" << std::endl;
    } else {
        std::cout << "viso_ is initialized." << std::endl;
    }

}

void
VISOPose::setExtrinsics(const std::vector<double>& I_q_C,
                        const std::vector<double>& I_p_C)
{
    Eigen::Vector3d p(I_p_C[0], I_p_C[1], I_p_C[2]);

    std::cout<< "inside setExtrinsics" << std::endl;


    Eigen::Quaterniond q;
    q.x() = I_q_C[0];
    q.y() = I_q_C[1];
    q.z() = I_q_C[2];
    q.w() = I_q_C[3];
    q.normalize();

    // std::cout<<"Quat is set \n";

    I_T_C_.block<3, 3>(0, 0) = q.toRotationMatrix();
    I_T_C_.block<3, 1>(0, 3) = p;

    C_T_I_ = I_T_C_.inverse();
}

void
VISOPose::tryProcessNextImages(boost::shared_ptr<cv::Mat> img0_data, boost::shared_ptr<cv::Mat> img1_data)
{
    if (img0_data == nullptr || img1_data == nullptr) {
        std::cerr << "Failed to load images "<< std::endl;
        return ;
    }

    // process up to the next keyframe time

    // auto msg0 = img0_queue_.front();
    // auto msg1 = img1_queue_.front();

    // ROS_INFO_STREAM("Processing VISO images, t0 = "
    //                 << msg0->header.stamp << ", t1 = " <<
    //                 msg1->header.stamp);
    // ROS_INFO_STREAM("Next key time = " << key_msg->header.stamp);

    // img0_queue_.pop_front();
    // img1_queue_.pop_front();
    // key_queue_.pop_front(); // do not pop keyframe yet, we might not have
    // reached it with intermediate images

    // cv::Mat img0 = cv_bridge::toCvShare(msg0, "mono8")->image;
    // cv::Mat img1 = cv_bridge::toCvShare(msg1, "mono8")->image;

    // cv::namedWindow("left");
    // cv::namedWindow("right");

    cv::Mat img0 = *img0_data;
    cv::Mat img1 = *img1_data;

    // ROS_INFO_STREAM("img0 is " << img0_raw.cols << " x " << img0_raw.rows);
    // ROS_INFO_STREAM("img1 is " << img1_raw.cols << " x " << img1_raw.rows);

    std::cout<<" Here: 1\n";

    // make sure we have an 8 bit grayscale image
    if (img0.channels() > 1) {
        cv::cvtColor(img0, img0, cv::COLOR_BGR2GRAY);
    }

    if (img1.channels() > 1) {
        cv::cvtColor(img1, img1, cv::COLOR_BGR2GRAY);
    }

    if (img0.type() != CV_8UC1) {
        img0.convertTo(img0, CV_8UC1);
    }

    if (img1.type() != CV_8UC1) {
        img1.convertTo(img1, CV_8UC1);
    }
    std::cout<<" Here: 2\n";
    int32_t dims[] = { img0.cols, img0.rows, img0.cols };

    // Eigen::Matrix4d fullp_eigen;

    // cv::waitKey(0);

    // ROS_INFO_STREAM("Dims = [" << dims[0] << ", " << dims[1] << ", " <<
    // dims[2]
    // << "]");

    uint8_t* img0_data_u8 =
      (uint8_t*)malloc(img0.rows * img0.cols * sizeof(uint8_t));
    uint8_t* img1_data_u8 =
      (uint8_t*)malloc(img0.rows * img0.cols * sizeof(uint8_t));

    for (size_t i = 0; i < img0.rows; ++i) {
        memcpy(
          img0_data_u8 + i * img0.cols, img0.ptr(i), img0.cols * sizeof(uint8_t));
        memcpy(
          img1_data_u8 + i * img1.cols, img1.ptr(i), img1.cols * sizeof(uint8_t));
    }
    std::cout<<" Here: 3\n";
    
    // copy back to image for debug test
    // cv::Mat img0_test(img0.rows, img0.cols, CV_8UC1);
    // cv::Mat img1_test(img0.rows, img0.cols, CV_8UC1);
    // for (size_t i = 0; i < img0.rows; ++i) {
    //     memcpy(img0_test.ptr(i), img0.ptr(i), img0.cols * sizeof(uint8_t));
    //     memcpy(img1_test.ptr(i), img1.ptr(i), img1.cols * sizeof(uint8_t));
    // }

    // cv::imshow("left", img0_test);
    // cv::imshow("right", img1_test);

    if (!initialized_) {
        // "not initialized" = this is the first image we process
        // can't get a pose estimate or anything (obviously) so just give to
        // viso and set up times etc
        viso_->process(img0_data_u8, img1_data_u8, dims, false);
        initialized_ = true;

        // free(img0_data);
        // free(img1_data);
        return;
    }
    std::cout<<" Here: 4\n";

    if (viso_->process(img0_data_u8, img1_data_u8, dims, false)) {
        // on success, update current pose
        // pose_ = pose_ * Matrix::inv(viso_->getMotion());
        // full_pose_ = full_pose_ * Matrix::inv(viso_->getMotion());

        // output some statistics
        // double num_matches = viso_->getNumberOfMatches();
        // double num_inliers = viso_->getNumberOfInliers();
        // std::cout << ", Matches: " << num_matches;
        // std::cout << ", Inliers: " << 100.0*num_inliers/num_matches << " %"
        // << ", Current pose: " << std::endl; std::cout <<
        // viso_->getPoseEstimate() << std::endl << std::endl;

        // std::cout << "Relative covariance: " << std::endl;
        // std::cout << viso_->getCovariance() << std::endl << std::endl;

        // std::cout << "Full Pose Covariance: " << std::endl;
        // std::cout << viso_->getPoseCovariance() << std::endl << std::endl;
    } else {
        std::cout<<" ... viso failed!";
    }

    // for (size_t i = 0; i < 4; ++i) {
    // 	for (size_t j = 0; j < 4; ++j) {
    // 		fullp_eigen(i,j) = full_pose_.val[i][j];
    // 	}
    // }

    // publishTransform(fullp_eigen);

    if (true) { // publish intermediate poses
        Eigen::Matrix4d relp = viso_->getPoseEstimate();
        out_file_ << frame_num_ << " ";
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                out_file_ << relp(i, j) << " ";
            }
        }
        out_file_ << std::endl;
        frame_num_++;
    }

    Eigen::Matrix4d pose = viso_->getPoseEstimate();
    Eigen::MatrixXd P = viso_->getPoseCovariance();

    // pose is of the *camera frame* now, C_T_C0
    // change to the body frame and invert to get our desired G_T_I

    pose = (I_T_C_ * pose * C_T_I_).inverse().eval();

    Eigen::Quaterniond q(pose.block<3, 3>(0, 0));
    Eigen::Vector3d p = pose.block<3, 1>(0, 3);

    // // Build odometry message
    // nav_msgs::Odometry msg;
    // msg.header = msg0->header;
    // // msg.header.stamp = ros::Time::now();
    // msg.header.frame_id = "odom";

    // msg.pose.pose.position.x = p(0);
    // msg.pose.pose.position.y = p(1);
    // msg.pose.pose.position.z = p(2);

    // msg.pose.pose.orientation.x = q.x();
    // msg.pose.pose.orientation.y = q.y();
    // msg.pose.pose.orientation.z = q.z();
    // msg.pose.pose.orientation.w = q.w();

    // cov is row-major
    // for (int i = 0; i < 6; ++i) {
    //     for (int j = 0; j < 6; ++j) {
    //         msg.pose.covariance[j + 6 * i] = P(i, j);
    //     }
    // }

    // relp_pub_.publish(msg);

    // viso_->resetPose();

    /*
            std::cout << "pose difference within viso:" << std::endl;
            for (int i = 0; i < 3; ++i) {
                    std::cout << "|";
                    for (int j = 0; j < 3; ++j) {
                            std::cout << relp(i,j) << (j < 2 ? ", " : "");
                    }
                    std::cout << "|" << std::endl;
            }
            std::cout << "[" << std::endl;
            for (int j = 0; j < 3; ++j) {
                    std::cout << relp(j, 3) << (j < 2 ? ", " : "");
            }
            std::cout << "]';" << std::endl << std::endl;
    */

    // svo_msgs::RelPose msg;
    // for (int i = 0; i < 4; ++i) {
    // 	for (int j = 0; j < 4; ++j) {
    // 		msg.relpose[4*i + j] = relp(i,j);
    // 	}
    // }

    // for (int i = 0; i < 6; ++i) {
    // 	for (int j = 0; j < 6; ++j) {
    // 		msg.cov[6*i + j] = P(i,j);
    // 	}
    // }

    // msg.header.stamp = msg0->header.stamp;

    // relp_pub_.publish(msg);

    // visoMatrixToEigen(pose_, last_key_pose_);

    free(img0_data_u8);
    free(img1_data_u8);
}

// void
// VISOPose::publishTransform(const Eigen::Matrix4d& pose)
// {
//     static tf::TransformBroadcaster br;
//     tf::Transform transform;
//     transform.setOrigin(tf::Vector3(pose(0, 3), pose(1, 3), pose(2, 3)));

//     Eigen::Vector4d q_vec = rot2quat(pose.block<3, 3>(0, 0).transpose());
//     tf::Quaternion q(q_vec(0), q_vec(1), q_vec(2), q_vec(3));
//     transform.setRotation(q);

//     br.sendTransform(
//       tf::StampedTransform(transform, ros::Time::now(), "world", "camera"));

//     geometry_msgs::Point pt;
//     pt.x = pose(0, 3);
//     pt.y = pose(1, 3);
//     pt.z = pose(2, 3);
//     trajectory_.push_back(pt);

//     visualization_msgs::Marker traj_msg;

//     traj_msg.header.frame_id = "world";
//     traj_msg.header.stamp = ros::Time::now();
//     traj_msg.ns = "trajectory";
//     traj_msg.action = visualization_msgs::Marker::ADD;
//     traj_msg.pose.orientation.w = 1.0;

//     traj_msg.id = 1;

//     traj_msg.type = visualization_msgs::Marker::LINE_STRIP;
//     traj_msg.scale.x = 1.0;
//     traj_msg.color.b = 1.0;
//     traj_msg.color.a = 1.0;

//     for (auto& p : trajectory_) {
//         traj_msg.points.push_back(p);
//     }

//     marker_pub_.publish(traj_msg);
// }

