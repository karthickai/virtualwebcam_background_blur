//
// Created by karthick on 31/08/20.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <gst/pbutils/pbutils.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/graph/default_device.h>

tensorflow::Tensor Mat2Tensor(cv::Mat &img, float normal = 1 / 255.0) {

    tensorflow::Tensor image_input = tensorflow::Tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(
            {1, img.size().height, img.size().width, img.channels()}));

    float *tensor_data_ptr = image_input.flat<float>().data();
    cv::Mat fake_mat(img.rows, img.cols, CV_32FC(img.channels()), tensor_data_ptr);
    img.convertTo(fake_mat, CV_32FC(img.channels()));

    fake_mat *= normal;

    return image_input;

}

int main() {

    std::cout << "OpenCV Version " << CV_VERSION << std::endl;
    std::cout << "gStreamer Version " << GST_VERSION_MAJOR << "." << GST_VERSION_MINOR << "." << GST_VERSION_MICRO
              << std::endl;

    // Load the protobuf graph
    tensorflow::GraphDef graph_def;
    std::string graph_path = "../bin/mobel.pb";
    tensorflow::Status status = tensorflow::ReadBinaryProto(tensorflow::Env::Default(), graph_path, &graph_def);
    if (!status.ok()) {
        std::cerr << status.ToString() << std::endl;
        return 1;
    } else {
        std::cout << "Load graph protobuf successfully" << std::endl;
    }
    for (int i = 0; i < graph_def.node_size(); ++i) {
//        auto node = graph_def.node(i);
//        std::cout << node.device() << std::endl;

    }
    tensorflow::SessionOptions opts;
//    tensorflow::graph::SetDefaultDevice("/GPU:0", &graph_def);
    opts.config.mutable_gpu_options()->set_per_process_gpu_memory_fraction(0.5);
    opts.config.mutable_gpu_options()->set_allow_growth(true);

    tensorflow::Session *session;
    status = tensorflow::NewSession(opts, &session);
    if (!status.ok()) {
        std::cout << status.ToString() << "\n";
        return 1;
    }

    // Add the graph to the session
    status = session->Create(graph_def);
    if (!status.ok()) {
        std::cerr << status.ToString() << std::endl;
        return 1;
    } else {
        std::cout << "Add graph to session successfully" << std::endl;
    }


    static const std::string windowName = "Virtual WebCam";
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    cv::VideoCapture cap;
    cap.open(std::string("/dev/video0"));
    cv::Mat frame, blob;
    int output_stride = 16;
    long frameCounter = 0;

    std::time_t timeBegin = std::time(0);
    int tick = 0;

    while (cv::waitKey(1) < 0) {
        cap >> frame;
        if (frame.empty()) {
            cv::waitKey();
            break;
        }
        auto frame_width = frame.size().width;
        auto frame_height = frame.size().height;
        auto target_width = std::floor(int(frame_width) / output_stride) * output_stride + 1;
        auto target_height = std::floor(int(frame_height) / output_stride) * output_stride + 1;
        auto target_size = cv::Size(target_width, target_height);
        cv::Mat dst;
        cv::resize(frame, dst, target_size);
        std::cout << "Target Width & Height " << target_width << " " << target_height << std::endl;
        auto width_resolution = int((target_width - 1) / output_stride) + 1;
        auto height_resolution = int((target_height - 1) / output_stride) + 1;
        std::cout << "Target Resolution Width & Height " << width_resolution << " " << height_resolution << std::endl;
        tensorflow::Tensor img = Mat2Tensor(dst, 1);
        std::cout << img.shape() << std::endl;
        // Setup inputs and outputs:
        std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
                {"sub_2:0", img}
        };
        std::vector<std::string> output_tensor = {
                "resnet_v1_50/displacement_bwd_2/BiasAdd:0", "resnet_v1_50/displacement_fwd_2/BiasAdd:0",
                "float_heatmaps:0", "float_long_offsets:0", "float_short_offsets:0", "float_part_heatmaps:0",
                "float_segments:0", "float_part_offsets:0"
        };

        std::vector<std::string> output_tensor_custom = {"float_segments:0"};
        // The session will initialize the outputs
        std::vector<tensorflow::Tensor> outputs;
        // Run the session, evaluating our "c" operation from the graph
        status = session->Run(inputs, output_tensor_custom, {}, &outputs);
        if (!status.ok()) {
            std::cerr << status.ToString() << std::endl;
            return 1;
        } else {
            std::cout << "Run session successfully" << std::endl;
        }
        std::cout << "output " << outputs.size() << std::endl;
        // Segmentation MASk
        auto segmentation_threshold = 0.7;
//        auto segmentScores = tensorflow
//        sigmoid(segments)
//        mask = tf.math.greater(segmentScores, tf.constant(segmentation_threshold))
//        print('maskshape', mask.shape)
//        segmentationMask = tf.dtypes.cast(mask, tf.int32)
//        segmentationMask = np.reshape(
//                segmentationMask, (segmentationMask.shape[0], segmentationMask.shape[1]))
        cv::imshow(windowName, dst);

//        cv::imwrite("simple.jpg", frame);
    }
    session->Close();

    return 0;
}