#include <opencv2/opencv.hpp>
#include <iostream>
#include <string> 
#include <fstream>
#include <sstream>
#include "main.h"

int main(int argc, char** argv)
{

    // Commandline arguments
    if (argc != 4 && argc != 7 && argc != 9)
    {
        std::cerr << "Usage: " << argv[0] << " IMAGE1 IMAGE2 OUTPUT_NAME" << std::endl;
        std::cerr << "or" << std::endl;
        std::cerr << "Usage: " << argv[0] << " IMAGE1 IMAGE2 OUTPUT_NAME FOCAL_LENGTH BASELINE DMIN" << std::endl;
        std::cerr << "or" << std::endl;
        std::cerr << "Usage: " << argv[0] << " IMAGE1 IMAGE2 OUTPUT_NAME FOCAL_LENGTH BASELINE DMIN WINDOW_SIZE WEIGHT" << std::endl;
        return 1;
    }

    cv::Mat image1 = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    cv::Mat image2 = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);
    const std::string output_file = argv[3];
    const double scale = 1;
    double focal_length, baseline, weight;
    int window_size, dmin;

    if (argc == 4)
    {
        std::cout << "Using defualt focal length (3740), default baseline (160), default dmin (67), default window size (5), default weight (500)" << std::endl;
        focal_length = 3740;
        baseline = 160;
        window_size = 5;
        weight = 500;
        dmin = 67;
    }
    else if (argc == 6)
    {
        std::cout << "Using default window size (5), default weight (500), default dmin (67)" << std::endl;
        focal_length = std::atof(argv[4]);
        baseline = std::atof(argv[5]);
        dmin = std::atoi(argv[6]);
        window_size = 5;
        weight = 500;
    }
    else
    {
        focal_length = std::atof(argv[4]);
        baseline = std::atof(argv[5]);
        dmin = std::atoi(argv[6]);
        window_size = std::atoi(argv[7]);
        weight = std::atof(argv[8]);
    }


    if (!image1.data)
    {
        std::cerr << "No image1 data" << std::endl;
        return EXIT_FAILURE;
    }

    if (!image2.data)
    {
        std::cerr << "No image2 data" << std::endl;
        return EXIT_FAILURE;
    }

    if (image1.size().height != image2.size().height || image1.size().width != image2.size().width)
    {
        std::cerr << "The two images must be of the same size" << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "------------------ Parameters -------------------" << std::endl;
    std::cout << "focal_length = " << focal_length << std::endl;
    std::cout << "baseline = " << baseline << std::endl;
    std::cout << "disparity added due to image cropping = " << dmin << std::endl;
    std::cout << "window_size = " << window_size << std::endl;
    std::cout << "occlusion weights = " << weight << std::endl;
    std::cout << "scaling of disparity images to show = " << scale << std::endl;
    std::cout << "output filename = " << argv[3] << std::endl;
    std::cout << "-------------------------------------------------" << std::endl;

    // Reconstruction
    // Naive disparity image
    cv::Mat naive_disparities = cv::Mat::zeros(image1.size().height - window_size + 1, image1.size().width - window_size + 1, CV_8UC1);
    cv::Mat dp_disparities = cv::Mat::zeros(image1.size().height - window_size + 1, image1.size().width - window_size + 1, CV_8UC1);

    StereoEstimation_Naive(window_size, image1, image2, naive_disparities, scale);
    StereoEstimation_Dynamic(window_size, weight, image1, image2, dp_disparities, scale);

    // Output //
    // reconstruction
    Disparity2PointCloud(output_file, dp_disparities, baseline, focal_length, dmin, scale);

    // save / display images
    std::stringstream out1;
    out1 << output_file << "_naive.png";
    cv::imwrite(out1.str(), naive_disparities);

    std::stringstream out2;
    out2 << output_file << "_dp.png";
    cv::imwrite(out2.str(), dp_disparities);

    cv::namedWindow("Naive", cv::WINDOW_AUTOSIZE);
    cv::imshow("Naive", naive_disparities);

    cv::namedWindow("Dynamic", cv::WINDOW_AUTOSIZE);
    cv::imshow("Dynamic", dp_disparities);

    cv::waitKey(0);

    return 0;
}

void StereoEstimation_Naive(int window_size, cv::Mat& image1, cv::Mat& image2, cv::Mat& naive_disparities, double scale)
{
    int half_window_size = window_size / 2;
    // loop through rows, where the window can be placed with the pixel in the middle
    for (int i = half_window_size; i - half_window_size + window_size <= image1.size().height; ++i)
    {
        std::cout << "Calculating disparities for the naive approach... "
            << std::ceil(((i) /(double)(image1.size().height - window_size)) * 100) << "%\r"
            << std::flush;

        // in a row, loop over columns, where the window can be placed with the pixel in the middle
        for (int j = half_window_size; j - half_window_size + window_size <= image1.size().width; ++j)
        {
            int min_ssd = INT_MAX;
            int disparity = 0;
            cv::Mat part1(image1, cv::Rect(j - half_window_size, i - half_window_size, window_size, window_size));
            // consider ALL possible translations, where the window is valid
            // we could nwrrow this down using dmin, but that would assume, that image1 and image2 is always ordered the same
            for (int d = -1 * j + half_window_size; j + d - half_window_size + window_size < image1.size().width; ++d)
            {
                cv::Mat part2(image2, cv::Rect(j + d - half_window_size, i - half_window_size, window_size, window_size));
                int ssd = cv::norm(part1, part2, cv::NORM_L2SQR);

                if (ssd < min_ssd)
                {
                    min_ssd = ssd;
                    disparity = d;
                }
            }
            naive_disparities.at<uchar>(i - half_window_size, j - half_window_size) = std::abs(disparity) * scale;
        }
    }

    std::cout << "Calculating disparities for the naive approach... Done.\r" << std::flush;
    std::cout << std::endl;
}

void StereoEstimation_Dynamic(int window_size, double weight, cv::Mat& image1, cv::Mat& image2, cv::Mat& dp_disparities, double scale)
{
    int half_window_size = window_size / 2;
    int elements_in_a_row = image1.size().width - window_size + 1;
    double num_elements_in_window = window_size * window_size;
    cv::Mat C;
    cv::Mat M;
    // loop through rows, where the window can be placed with the pixel in the middle
    for (int row = half_window_size; row - half_window_size + window_size <= image1.size().height; ++row)
    {
        std::cout << "Calculating disparities for the dynamic approach... "
            << std::ceil(((row) / (double)(image1.size().height - window_size)) * 100) << "%\r"
            << std::flush;
        // initialize cost and choice matrices. cost matrix will be padded with an "artificial" first row and column
        C = cv::Mat::zeros(elements_in_a_row + 1, elements_in_a_row + 1, CV_64FC1);
        // in the first column, we want to have values: 0, 0, weight, 2 * weight, etc
        for (int r = 0; ++r < C.size().height; C.at<double>(r, 0) = (r - 1) * weight);
        // in the first row, we want to have values: 0, 0, weight, 2 * weight, etc
        for (int c = 0; ++c < C.size().width; C.at<double>(0, c) = (c - 1) * weight);
        M = cv::Mat::zeros(elements_in_a_row, elements_in_a_row, CV_8UC1);

        // now we can fill the matrices
        for (int i = 0; i < elements_in_a_row; ++i)
        {
            // get image patch from first image, i (j also) can be interpreted as the starting x coordinate of the window
            cv::Mat part1(image1, cv::Rect(i, row - half_window_size, window_size, window_size));
            for (int j = 0; j < elements_in_a_row; ++j)
            {
                // because of the padding, we want to fill the element at i+1,j+1 so the previous indicies are i and j
                double occ_1 = C.at<double>(i, j + 1) + weight;
                double occ_2 = C.at<double>(i + 1, j) + weight;
                cv::Mat part2(image2, cv::Rect(j, row - half_window_size, window_size, window_size));
                double match = C.at<double>(i, j) + cv::norm(part1, part2, cv::NORM_L2SQR) / num_elements_in_window;
                if (match < occ_1 && match < occ_2)
                {
                    // in this row, pixel i of image1, and pixel j of image2 match
                    C.at<double>(i + 1, j + 1) = match;
                    M.at<uchar>(i, j) = 0;
                }
                else if (occ_1 < occ_2)
                {
                    // in this row, pixel i of image1 is occluded (does not match a pixel in the row of image2)
                    C.at<double>(i + 1, j + 1) = occ_1;
                    M.at<uchar>(i, j) = 1;
                }
                else
                {
                    // in this row, pixel j of image2  is occluded (does not match a pixel in the row of image1)
                    C.at<double>(i + 1, j + 1) = occ_2;
                    M.at<uchar>(i, j) = 2;
                }
            }
        }

        // and now decode the path of the row. we want to get the disparity image corresponding to image1
        int i = elements_in_a_row - 1;
        int j = elements_in_a_row - 1;
        while (i >= 0 && j >= 0)
        {
            switch (M.at<uchar>(i, j))
            {
            case 0:
                // in this row, pixel i of image1, and pixel j of image2 match
                dp_disparities.at<uchar>(row - half_window_size, i) = cv::abs(i - j) * scale;
                --i;
                --j;
                break;
            case 1:
                // in this row, pixel i of image1, does not match a pixel in the row of image2
                // so we say, that we use the same dispartiy as with pixel i + 1
                if (i != elements_in_a_row - 1)
                {
                    dp_disparities.at<uchar>(row - half_window_size, i) = dp_disparities.at<uchar>(row - half_window_size, i + 1);
                }
                --i;
                break;
            case 2:
                // in this row, pixel j of image2, does not match a pixel in the row of image1
                --j;
                break;
            }
        }
        while (i >= 0)
        {
            dp_disparities.at<uchar>(row - half_window_size, i) = dp_disparities.at<uchar>(row - half_window_size, i + 1);
            --i;
        }
    }

    std::cout << "Calculating disparities for the dynamic approach... Done.\r" << std::flush;
    std::cout << std::endl;
}

void Disparity2PointCloud(const std::string& output_file, cv::Mat& disparities, double baseline, double focal_length, int dmin, double scale)
{
    std::stringstream out3d;
    out3d << output_file << ".xyz";
    std::ofstream outfile(out3d.str());
    for (int i = 0; i < disparities.size().height; ++i)
    {
        std::cout << "Reconstructing 3D point cloud from disparities... "
            << std::ceil(((i) / static_cast<double>(disparities.size().height)) * 100)
            << "%\r" << std::flush;
        for (int j = 0; j < disparities.size().width; ++j)
        {
            int d = disparities.at<uchar>(i, j) / scale + dmin;
            if (d == 0)
            {
                continue;
            }
            
            const double Z = baseline * focal_length / d;
            const double X = (baseline * (j + j + d)) / (2 * d);
            const double Y = baseline * i / d;
            
            outfile << X << " " << Y << " " << Z << std::endl;
        }
    }
    std::cout << "Reconstructing 3D point cloud from disparities... Done.\r" << std::flush;
    std::cout << std::endl;
}
