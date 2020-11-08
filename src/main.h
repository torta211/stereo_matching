#pragma once

void StereoEstimation_Naive(int window_size, cv::Mat& image1, cv::Mat& image2, cv::Mat& naive_disparities, double scale);

void StereoEstimation_Dynamic(int window_size, double weight, cv::Mat& image1, cv::Mat& image2, cv::Mat& dp_disparities, double scale);

void Disparity2PointCloud(const std::string& output_file, cv::Mat& disparities, double baseline, double focal_length, int dmin, double scale);
