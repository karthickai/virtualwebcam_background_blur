//
// Created by karthick on 31/08/20.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <gst/pbutils/pbutils.h>
#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/public/session.h>

int main() {

    std::cout << "OpenCV Version " << CV_VERSION << std::endl;
    std::cout << "gStreamer Version " << GST_VERSION_MAJOR << "." << GST_VERSION_MINOR << "." << GST_VERSION_MICRO
              << std::endl;
    tensorflow::DeviceTypeString("GPU");
    tensorflow::Session *session;
    tensorflow::Status status = tensorflow::NewSession(tensorflow::SessionOptions(), &session);
    if (!status.ok()) {
        std::cout << status.ToString() << "\n";
        return 1;
    }

    // Load the protobuf graph
    tensorflow::GraphDef graph_def;
    std::string graph_path = "../bin/mobel.pb";
    status = tensorflow::ReadBinaryProto(tensorflow::Env::Default(), graph_path, &graph_def);
    if (!status.ok()) {
        std::cerr << status.ToString() << std::endl;
        return 1;
    } else {
        std::cout << "Load graph protobuf successfully" << std::endl;
    }

    // Add the graph to the session
    status = session->Create(graph_def);
    if (!status.ok()) {
        std::cerr << status.ToString() << std::endl;
        return 1;
    } else {
        std::cout << "Add graph to session successfully" << std::endl;
    }

    session->Close();

    static const std::string windowName = "Virtual WebCam";
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    cv::VideoCapture cap;
    cap.open(std::string("/dev/video0"));
    cv::Mat frame, blob;
    while (cv::waitKey(1) < 0) {
        cap >> frame;
        if (frame.empty()) {
            cv::waitKey();
            break;
        }
        cv::imshow(windowName, frame);
    }
    return 0;
}