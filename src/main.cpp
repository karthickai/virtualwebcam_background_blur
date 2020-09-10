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



    // Read image
    cv::Mat img, inp;
    img = cv::imread("../bin/simple.jpg", cv::IMREAD_COLOR);

    int rows = img.rows;
    int cols = img.cols;

    auto frame_width = img.size().width;
    auto frame_height = img.size().height;
    int output_stride = 16;
    auto target_width = std::floor(int(frame_width) / output_stride) * output_stride + 1;
    auto target_height = std::floor(int(frame_height) / output_stride) * output_stride + 1;
    cv::resize(img, inp, cv::Size(target_width, target_height));
    cv::cvtColor(inp, inp, cv::COLOR_BGR2RGB);

    std::cout << inp.size << std::endl;

    // Put image in Tensor
    std::vector<uint8_t > img_data;
    img_data.assign(inp.data, inp.data + inp.total() * inp.channels());
    Model model("../bin/mobel_old.pb");
    for (auto ops : model.get_operations()){
        std::cout << ops << std::endl;
    }
    Tensor outNames1{model, "float_heatmaps"};
    Tensor inpName{model, "sub_2"};

    inpName.set_data(img_data, {1, static_cast<long>(target_width), static_cast<long>(target_height), 3});

    model.run(inpName, {&outNames1});

    return 0;
}
