#include "opencv2/opencv.hpp"
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <algorithm>
#include <tuple>

constexpr float CONFIDENCE_THRESHOLD = 0.9;
constexpr float NMS_THRESHOLD = 0.4;
constexpr int NUM_CLASSES = 80;
const cv::Scalar colors[] = {
    {0, 255, 255},
    {255, 255, 0},
    {0, 255, 0},
    {255, 0, 0}};
const auto NUM_COLORS = sizeof(colors) / sizeof(colors[0]);

bool myfunction(std::tuple<cv::Rect, float> i, std::tuple<cv::Rect, float> j) { return (std::get<1>(i) > std::get<1>(j)); }

void NonMaxSupression(std::vector<std::tuple<cv::Rect, float>> (&all_detection_boxes)[NUM_CLASSES], float IOUThreshold)
{

    std::vector<cv::Rect> filtered_detection_boxes[NUM_CLASSES];
    std::vector<float> filtered_detection_scores[NUM_CLASSES];

    for (int all_detection_boxes_index = 0; all_detection_boxes_index < NUM_CLASSES; all_detection_boxes_index++)
    {

        std::vector<std::tuple<cv::Rect, float>> &class_detection_boxes = all_detection_boxes[all_detection_boxes_index];
        std::sort(class_detection_boxes.begin(), class_detection_boxes.end(), myfunction);
        if (class_detection_boxes.size() == 0)
        {
            continue;
        }

        int class_detection_boxes_index = 0;
        while (class_detection_boxes_index < class_detection_boxes.size())
        {
            cv::Rect most_confident_box = std::get<0>(class_detection_boxes[class_detection_boxes_index]);
            float most_confident_box_score = std::get<1>(class_detection_boxes[class_detection_boxes_index]);
            int filter_index = class_detection_boxes_index + 1;
            while (filter_index < class_detection_boxes.size())
            {

                cv::Rect detection_box = std::get<0>(class_detection_boxes[filter_index]);
                float detection_box_condfidence = std::get<1>(class_detection_boxes[filter_index]);
                if ((detection_box.x < most_confident_box.x) && (detection_box.x + detection_box.width < most_confident_box.x + most_confident_box.width) || (detection_box.x > most_confident_box.x + most_confident_box.width))
                {
                    filter_index++;
                    continue;
                }
                else if ((detection_box.y < most_confident_box.y) && (detection_box.y + detection_box.height < most_confident_box.y + most_confident_box.height) || (detection_box.y > most_confident_box.y + most_confident_box.height))
                {
                    filter_index++;
                    continue;
                }
                int x_overlap_min_boundary = std::min(detection_box.x, most_confident_box.x);
                int x_overlap_max_boundary = std::max(detection_box.x + detection_box.width, most_confident_box.x + most_confident_box.width);
                int max_x_overlap = x_overlap_max_boundary - x_overlap_min_boundary;
                max_x_overlap = max_x_overlap - (detection_box.x - x_overlap_min_boundary);
                max_x_overlap = max_x_overlap - (x_overlap_max_boundary - detection_box.x + detection_box.width);

                int y_overlap_min_boundary = std::min(detection_box.y, most_confident_box.y);
                int y_overlap_max_boundary = std::max(detection_box.y + detection_box.height, most_confident_box.y + most_confident_box.height);
                int max_y_overlap = y_overlap_max_boundary - y_overlap_min_boundary;
                max_y_overlap = max_y_overlap - (detection_box.y - y_overlap_min_boundary);
                max_y_overlap = max_y_overlap - (y_overlap_max_boundary - detection_box.y + detection_box.height);

                int overlap_area = max_x_overlap * max_y_overlap;
                float overlap_percentage = overlap_area / detection_box.area();
                if (overlap_percentage > IOUThreshold)
                {
                    class_detection_boxes.erase(class_detection_boxes.begin() + filter_index);
                }
                else
                {
                    filter_index++;
                }
            }
            class_detection_boxes_index++;
        }
    }
}

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
    neural_net.setPreferableBackend(cv::dnn::DNN_TARGET_CPU);
    neural_net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    auto output_names = neural_net.getUnconnectedOutLayersNames();
    cv::Mat frame;
    cv::Mat blob;
    std::vector<cv::Mat> neural_net_output;

    std::vector<int> class_indices[NUM_CLASSES];

    std::vector<std::tuple<cv::Rect, float>> all_detections[NUM_CLASSES];

    for (;;)
    {
        if (!cap.read(frame))
        {
            std::cout << "cant read video file...\n";
            break;
        }
        auto IMAGE_WIDTH = frame.cols;
        auto IMAGE_HEIGHT = frame.rows;
        float detection_box_x;
        float detection_box_y;
        float detection_box_width;
        float detection_box_height;
        // auto total_start = std::chrono::steady_clock::now();
        cv::dnn::blobFromImage(frame, blob, 0.00392, cv::Size(608, 608), cv::Scalar(), true, false, CV_32F);
        neural_net.setInput(blob);

        // auto dnn_start = std::chrono::steady_clock::now();
        neural_net.forward(neural_net_output, output_names);
        // auto dnn_end = std::chrono::steady_clock::now();

        for (auto &detection : neural_net_output)
        {
            const auto num_boxes = detection.rows;
            for (int i = 0; i < num_boxes; i++)
            {
                float x = detection.at<float>(i, 0) * IMAGE_WIDTH;
                auto y = detection.at<float>(i, 1) * IMAGE_HEIGHT;
                auto width = detection.at<float>(i, 2) * IMAGE_WIDTH;
                auto height = detection.at<float>(i, 3) * IMAGE_HEIGHT;

                cv::Rect rect(x - width / 2, y - height / 2, width, height);
                for (int c = 0; c < NUM_CLASSES; c++)
                {
                    auto confidence = *detection.ptr<float>(i, 5 + c);
                    if (confidence >= CONFIDENCE_THRESHOLD)
                    {
                        std::tuple<cv::Rect, float> class_detection = std::make_tuple(rect, confidence);
                        all_detections[c].push_back(class_detection);
                    }
                }
            }
        }

        NonMaxSupression(all_detections, NMS_THRESHOLD);

        for (int c = 0; c < NUM_CLASSES; c++)
        {
            auto &class_detections = all_detections[c];
            // std::vector<float> class_detection_scores = detection_scores[c];

            for (auto &class_detection : class_detections)
            {
                const auto color = colors[c % NUM_COLORS];
                cv::rectangle(frame, cv::Point(std::get<0>(class_detection).x, std::get<0>(class_detection).y), cv::Point(std::get<0>(class_detection).x + std::get<0>(class_detection).width, std::get<0>(class_detection).y + std::get<0>(class_detection).height), color, 3);
            }
            class_detections.clear();
        }
        // auto total_end = std::chrono::steady_clock::now();

        cv::imshow("Detections", frame);
        if (cv::waitKey(30) == 27)
        {
            break;
        }
    }
    return 0;
}
