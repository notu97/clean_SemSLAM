#include <iostream>
#include <fstream>
#include <string>

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "libviso2/viso_math.h"
#include "viso_pose/VISOPose.h"

#include <chrono>
#include <future>

boost::shared_ptr<cv::Mat> readImage(const std::string& filename, bool img0, bool save_images) {
    boost::shared_ptr<cv::Mat> img = boost::make_shared<cv::Mat>(cv::imread(filename));
    if (img->empty()) {
        std::cerr << "Failed to load image: " << filename << std::endl;
        return nullptr;
    }

    std::cout << "Reading image: " << filename << std::endl;

    if (save_images)
    {
        size_t pos = filename.find_last_of("/\\");
        std::string store_filename = filename.substr(pos + 1);
    
        // Write the image to a file for debugging purposes
        std::string folder_name = img0 ? "image_2" : "image_3";
        std::string debug_folder = "/dataset/" + folder_name + "/" + store_filename;
    
        cv::imwrite(debug_folder, *img);
        std::cout << "Wrote debug image: " << debug_folder << std::endl;
    }

    return img;
}

int
main(int argc, char* argv[])
{
    if (argc < 2) {
        std::cerr << "Usage: ./semSLAM <KITTI_DIR> <save_images>" << std::endl;
        return 1;
    }

    std::string kitti_dir = argv[1];
    double rate = 1.0;
    double t_start = 0.0;
    double t_end = std::numeric_limits<double>::infinity();
    double delay_secs = 1.0;
    bool save_images = argv[2] == std::string("true") ? true : false;


    
    std::string cam_ts_filename = kitti_dir + "/times.txt";
    std::ifstream cam_ts_file(cam_ts_filename);

    if (!cam_ts_file) {
        std::cerr << "Unable to open file " << cam_ts_filename << std::endl;
        return 1;
    }

    // read camera times
    std::vector<double> cam_ts;
    while (cam_ts_file.good()) {
        double t;
        cam_ts_file >> t;
        cam_ts.push_back(t / rate);
    }

    std::cout << cam_ts.size() << " camera msmts\n";


    // read calibration information
    std::string cam_calib_filename = kitti_dir + "/calib.txt";
    std::ifstream cam_calib_file(cam_calib_filename);

    if (!cam_calib_file) {
        std::cerr << "Unable to open file " << cam_calib_filename << std::endl;
        return 1;
    }

    std::string ln, token;
    CameraInfo info0_msg, info1_msg;
    info0_msg.height = 370;
    info0_msg.width = 1226;
    info0_msg.distortion_model = "plumb_bob";
    info0_msg.binning_x = 1;
    info0_msg.binning_y = 1;

    while (std::getline(cam_calib_file, ln)) {
        std::stringstream ss(ln);
        ss >> token;

        if (token.find("P2") == 0) {
            for (int i = 0; i < 12; ++i) {
                ss >> info0_msg.P[i];
            }
        } else if (token.find("P3") == 0) {
            for (int i = 0; i < 12; ++i) {
                ss >> info1_msg.P[i];
            }
        }
    }

    std::cout << "Camera 0 P = {";
    for (double x : info0_msg.P) {
        std::cout << x << ", ";
    }
    std::cout << "}" << std::endl;
    std::cout << "Camera 1 P = {";
    for (double k : info1_msg.P) {
        std::cout << k << ", ";
    }
    std::cout << "}" << std::endl;

    // Initialize Libviso 
    double px_sigma = 4.0;
    VISOPose viso(px_sigma);

    std::cout << "Setting VISOPose parameters..." << std::endl;
    viso.setVisoParams(info0_msg, info1_msg);

    std::cout << "Setting extrinsics..." << std::endl;
    std::vector<double> I_q_C = {-0.5, 0.5, -0.5, 0.5}; // Quaternion
    std::vector<double> I_p_C = {0.0, 0.0, 0.0}; // Position
    viso.setExtrinsics(I_q_C, I_p_C);


    std::string cam0_fmt_str = kitti_dir + "/image_2/%06d.png";
    std::string cam1_fmt_str = kitti_dir + "/image_3/%06d.png";
    size_t cam_next = 0;

    // Find first index
    while (cam_ts[cam_next] * rate < t_start)
        cam_next++;
    
    
    // Record start time
    auto start = std::chrono::high_resolution_clock::now();

    while(cam_next < cam_ts.size() - 1) {

        if (cam_ts[cam_next] * rate > t_end) {
            break;
        }
        std::cout << "Processing camera index: " << cam_next << std::endl;

        // Asynchronously read images
        char fname0[1024];
        sprintf(fname0, cam0_fmt_str.c_str(), cam_next);
        std::future<boost::shared_ptr<cv::Mat>> img0_data_future = std::async(std::launch::async, readImage,fname0, true, save_images);

        char fname1[1024];
        sprintf(fname1, cam1_fmt_str.c_str(), cam_next);
        std::future<boost::shared_ptr<cv::Mat>> img1_data_future = std::async(std::launch::async, readImage,fname1, false, save_images);

        auto img0_data = img0_data_future.get();  // resolve the future
        auto img1_data = img1_data_future.get();  // resolve the future

        if (img0_data == nullptr || img1_data == nullptr) {
            std::cerr << "Failed to load images at index " << cam_next << std::endl;
            return 1;
        }

        viso.tryProcessNextImages(img0_data, img1_data);

        // if (img0.empty() || img1.empty()) {
        //     std::cerr << "Failed to load images at index " << cam_next << std::endl;
        //     return 1;
        // }
        
        cam_next++;
    }

    // Record end time
    auto end = std::chrono::high_resolution_clock::now();

    // Compute duration
    std::chrono::duration<double, std::milli> duration_ms = end - start;
    std::cout << "Execution time: " << duration_ms.count() << " ms" << std::endl;

    return 0;
}