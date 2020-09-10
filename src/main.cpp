//
// Created by karthick on 10/09/20.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include <tensorflow/c/c_api.h>

#include "Model.h"

int main() {
    std::cout << "OpenCV Version " << CV_VERSION << std::endl;
    std::cout << "TensorFlow library version " << TF_Version() << std::endl;

    Model model("../bin/mobel.pb");
    Tensor outNames1{model, "resnet_v1_50/displacement_bwd_2/BiasAdd:0"};
    Tensor inpName{model, "sub_2:0"};

    // Read image
    cv::Mat img, inp;
    img = cv::imread("../test.png", cv::IMREAD_COLOR);

    int rows = img.rows;
    int cols = img.cols;

    cv::resize(img, inp, cv::Size(300, 300));
    cv::cvtColor(inp, inp, cv::COLOR_BGR2RGB);

    // Put image in Tensor
    std::vector<uint8_t > img_data;
    img_data.assign(inp.data, inp.data + inp.total() * inp.channels());
    inpName.set_data(img_data, {1, 300, 300, 3});

    model.run(inpName, {&outNames1});

    return 0;
}
