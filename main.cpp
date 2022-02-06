#include "opencv2/opencv.hpp"
#include <opencv2/dnn.hpp>
#include <iostream>

constexpr float CONFIDENCE_THRESHOLD = 0;
constexpr float NMS_THRESHOLD = 0.4;
constexpr int NUM_CLASSES = 80;
const cv::Scalar colors[] = {
    {0, 255, 255},
    {255, 255, 0},
    {0, 255, 0},
    {255, 0, 0}};
const auto NUM_COLORS = sizeof(colors) / sizeof(colors[0]);

int main(int argc, char **argv)
{

    cv::VideoCapture cap("people_walking.mp4");
    if (!cap.isOpened())
    {
        std::cout << "Cannot open video file. \n";
        return 1;
    }
    // cv::namedWindow("Detections");
    auto neural_net = cv::dnn::readNet("yolov4.cfg", "yolov4.weights");
    neural_net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    neural_net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    auto output_names = neural_net.getUnconnectedOutLayersNames();
    cv::Mat frame;
    cv::Mat blob;
    std::vector<cv::Mat> detections;

    std::vector<int> class_indices[NUM_CLASSES];
    std::vector<cv::Rect> detection_boxes[NUM_CLASSES];
    std::vector<float> detection_scores[NUM_CLASSES];
    for (;;)
    {
        if (!cap.read(frame))
        {
            std::cout << "cant read video file...\n";
            break;
        }
        auto total_start = std::chrono::steady_clock::now();
        cv::dnn::blobFromImage(frame, blob, 0.00392, cv::Size(608, 608), cv::Scalar(), true, false, CV_32F);
        neural_net.setInput(blob);

        auto dnn_start = std::chrono::steady_clock::now();
        neural_net.forward(detections, output_names);
        auto dnn_end = std::chrono::steady_clock::now();

        for (auto &detection : detections)
        {
            const auto num_boxes = detection.rows;
            for (int i = 0; i < num_boxes; i++)
            {
                auto x = detection.at<float>(i, 0);
                auto y = detection.at<float>(i, 1);
                auto width = detection.at<float>(i, 2);
                auto height = detection.at<float>(i, 3);
                cv::Rect rect(x - width / 2, y - height / 2, width, height);

                for (int c = 0; c < NUM_CLASSES; c++)
                {
                    auto confidence = *detection.ptr<float>(i, 5 + c);
                    if (confidence >= CONFIDENCE_THRESHOLD)
                    {
                        detection_boxes[c].push_back(rect);
                        detection_scores[c].push_back(confidence);
                    }
                }
            }
        }

        for (int c = 0; c < NUM_CLASSES; c++)
        {
            for (size_t i = 0; i < class_indices[c].size(); ++i)
            {
                const auto color = colors[c % NUM_COLORS];
                auto detection_idx = class_indices[c][i];
                const auto& rect = detection_boxes[c][detection_idx];
                cv::rectangle(frame, cv::Point(rect.x, rect.y), cv::Point(rect.x + rect.width, rect.y + rect.height), color, 3);
            }
        }
        auto total_end = std::chrono::steady_clock::now();

        cv::imshow("Detections", frame);
        if (cv::waitKey(30) == 27)
        {
            break;
        }
    }

    return 0;
}