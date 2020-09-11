//
// Created by karthick on 10/09/20.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <tensorflow/c/c_api.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/videoio/videoio_c.h>

#include "Model.h"

// OpenCV helper functions
cv::Mat convert_rgb_to_yuyv(cv::Mat input) {
    cv::Mat tmp;
    cv::cvtColor(input, tmp, CV_RGB2YUV);
    std::vector<cv::Mat> yuv;
    cv::split(tmp, yuv);
    cv::Mat yuyv(tmp.rows, tmp.cols, CV_8UC2);
    uint8_t *outdata = (uint8_t *) yuyv.data;
    uint8_t *ydata = (uint8_t *) yuv[0].data;
    uint8_t *udata = (uint8_t *) yuv[1].data;
    uint8_t *vdata = (uint8_t *) yuv[2].data;
    for (unsigned int i = 0; i < yuyv.total(); i += 2) {
        uint8_t u = (uint8_t) (((int) udata[i] + (int) udata[i + 1]) / 2);
        uint8_t v = (uint8_t) (((int) vdata[i] + (int) vdata[i + 1]) / 2);
        outdata[2 * i + 0] = ydata[i + 0];
        outdata[2 * i + 1] = v;
        outdata[2 * i + 2] = ydata[i + 1];
        outdata[2 * i + 3] = u;
    }
    return yuyv;
}

int main() {
    std::cout << "OpenCV Version " << CV_VERSION << std::endl;
    std::cout << "TensorFlow library version " << TF_Version() << std::endl;
    int width = 640;
    int height = 480;


    cv::Mat bg = cv::imread("../bin/bg.jpg");
    cv::resize(bg, bg, cv::Size(width, height));

    cv::VideoCapture cap;
    cap.open(std::string("/dev/video0"));
    cv::Mat frame;
    cap.set(CV_CAP_PROP_FRAME_WIDTH,  width);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, height);


    // Model Setup
    Model model("../bin/mobel.pb");
//    for (auto ops : model.get_operations()) {
//        std::cout << ops << std::endl;
//    }
    Tensor outNames1{model, "float_segments"};
    Tensor inpName{model, "sub_2"};

    // initialize mask and square ROI in center
    cv::Rect roidim = cv::Rect((width - height) / 2, 0, height, height);
    cv::Mat mask = cv::Mat::ones(height, width, CV_8UC1);
    cv::Mat mroi = mask(roidim);

    // erosion/dilation element
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, 1));

    while (cv::waitKey(1) < 0) {
        cap >> frame;

        // Image to Tensor
        cv::Mat roi = frame(roidim);
        cv::Mat in_u8, ip;
        cv::resize(roi, in_u8, cv::Size(257, 257));
        in_u8.convertTo(ip,CV_32FC3,1.0/128.0,-1.0);
        std::vector<float> img_data;
        img_data.assign(ip.data, ip.data + ip.total() * ip.channels());
        const std::vector<std::int64_t> input_dims = {1, 257, 257, 3};
        inpName.set_data(img_data, input_dims);

        //inference
        model.run(inpName, {&outNames1});

        auto output = outNames1.convert_tensor_to_mat();
        cv::Mat ofinal(output.rows,output.cols,CV_8UC1);         // create Mat for small mask
        float* tmp = (float*)output.data;
        uint8_t* out = (uint8_t*)ofinal.data;

        // threshold probability
        for (unsigned int n = 0; n < output.total(); n++) {
            if (tmp[n] > 0.65) out[n] = 0; else out[n] = 255;
        }

        // denoise
//        cv::Mat tmpbuf;
//        cv::dilate(ofinal,tmpbuf,element);
//        cv::erode(tmpbuf,ofinal,element);

        // scale up into full-sized mask
        cv::resize(ofinal,mroi,cv::Size(frame.rows,frame.rows));

        // copy background over raw cam image using mask
        bg.copyTo(frame,mask);
        cv::imshow("output.png",frame);
    }


    // Read image
//    cv::Mat img, inp;
//    img = cv::imread("../bin/simple.jpg", cv::IMREAD_COLOR);


//    auto frame_width = img.size().width;
//    auto frame_height = img.size().height;
//    int output_stride = 16;
//    auto target_width = std::floor(int(frame_width) / output_stride) * output_stride + 1;
//    auto target_height = std::floor(int(frame_height) / output_stride) * output_stride + 1;
//    cv::resize(img, inp, cv::Size(257, 257));
//    cv::cvtColor(inp, inp, cv::COLOR_BGR2RGB);
//
//    std::cout << inp.size << std::endl;
//
//
//    // Put image in Tensor
//    std::vector<float> img_data;
//    img_data.assign(inp.data, inp.data + inp.total() * inp.channels());
//    std::cout << img_data.size() << std::endl;
//    const std::vector<std::int64_t> input_dims = {1, 257, 257, 3};
//    inpName.set_data(img_data, input_dims);
//    model.run(inpName, {&outNames1});
//    auto op = outNames1.get_data<float>();
//    auto opImg = outNames1.convert_tensor_to_mat();

//    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
//    img = convert_rgb_to_yuyv(img);



//    cv::Mat ofinal(opImg.rows, opImg.cols, CV_8UC1);
//    float *tmp = (float *) opImg.data;
//    uint8_t *out = (uint8_t *) ofinal.data;
//    for (unsigned int n = 0; n < opImg.total(); n++) {
//        // FIXME: hardcoded threshold
//        if (tmp[n] > 0.65) out[n] = 0; else out[n] = 255;
//    }
//
//    // denoise
//    cv::Mat tmpbuf;
//    cv::dilate(ofinal, tmpbuf, element);
//    cv::erode(tmpbuf, ofinal, element);
//    cv::resize(ofinal, mroi, cv::Size(img.rows, img.rows));
//    bg.copyTo(img, mask);
//
//    cv::Mat test;
//    cv::cvtColor(img,test,CV_YUV2BGR_YUYV);
//    cv::imshow("output.png",test);
//    cv::waitKey(0);
//
//    std::cout << op.size() << std::endl;
//    for( auto s : inpName.get_shape()){
//        std::cout << s << std::endl;
//    }
//    for (auto vp : op) {
////        std::cout << vp << std::endl;
//    }

    return 0;
}
