//
// Created by karthick on 31/08/20.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include <gst/pbutils/pbutils.h>

int main() {

    std::cout << "OpenCV Version " << CV_VERSION << std::endl;
    std::cout << "gStreamer Version " << GST_VERSION_MAJOR << "." << GST_VERSION_MINOR << "." << GST_VERSION_MICRO
              << std::endl;
    std::cout << "Hello World" << std::endl;
    return 0;
}