#include "opencv2/opencv.hpp"
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

constexpr float CONFIDENCE_THRESHOLD = 0.9;
constexpr float NMS_THRESHOLD = 0.4;
constexpr int NUM_CLASSES = 80;
const cv::Scalar colors[] = {
    {0, 255, 255},
    {255, 255, 0},
    {0, 255, 0},
    {255, 0, 0}};
const auto NUM_COLORS = sizeof(colors) / sizeof(colors[0]);

int FindHighestScoringDetectionBox(std::vector<cv::Rect> class_detection_boxes, std::vector<float> class_detection_scores)
{
    float highest_confidence_score = -INFINITY;
    int most_confident_box_index;

    //find highest scoring box
    int tes1 = class_detection_boxes.size();
    for(int class_detection_boxes_index = 0; class_detection_boxes_index < class_detection_boxes.size(); class_detection_boxes_index++)
    {
        cv::Rect detection_box = class_detection_boxes[class_detection_boxes_index];
        float detection_box_confidence_score = class_detection_scores[class_detection_boxes_index];
        if(detection_box_confidence_score > highest_confidence_score)
        {
            highest_confidence_score = detection_box_confidence_score;
            most_confident_box_index = class_detection_boxes_index;
        }
    }
    return most_confident_box_index;
}

void IOUFilter(cv::Rect most_confident_box, std::vector<cv::Rect> class_detection_boxes, std::vector<float> class_detection_scores, float IOUThreshHold, std::vector<cv::Rect> filtered_class_detection_boxes)
{
    int class_detection_boxes_index = 0;
    while(class_detection_boxes_index < class_detection_boxes.size())
    {
        cv::Rect detection_box = class_detection_boxes[class_detection_boxes_index];
        if((detection_box.x < most_confident_box.x) && (detection_box.x + detection_box.width < most_confident_box.x + most_confident_box.width) 
            || (detection_box.x > most_confident_box.x + most_confident_box.width))
        {
            class_detection_boxes_index++;
            continue;  
        }
        else if((detection_box.y < most_confident_box.y) && (detection_box.y + detection_box.height < most_confident_box.y + most_confident_box.height)
            || (detection_box.y > most_confident_box.y + most_confident_box.height))
        {
            class_detection_boxes_index++;
            continue;
        }
        int x_overlap_min_boundary = std::min(detection_box.x, most_confident_box.x);
        int x_overlap_max_boundary = std::max(detection_box.x + detection_box.width, most_confident_box.x + most_confident_box.width);
        int max_x_overlap = x_overlap_max_boundary - x_overlap_min_boundary;
        max_x_overlap = max_x_overlap - (detection_box.x - x_overlap_min_boundary);
        max_x_overlap = max_x_overlap  - (x_overlap_max_boundary - detection_box.x + detection_box.width);

        int y_overlap_min_boundary = std::min(detection_box.y, most_confident_box.y);
        int y_overlap_max_boundary = std::max(detection_box.y + detection_box.height, most_confident_box.y + most_confident_box.height);
        int max_y_overlap = y_overlap_max_boundary - y_overlap_min_boundary;
        max_y_overlap = max_y_overlap - (detection_box.y - y_overlap_min_boundary);
        max_y_overlap = max_y_overlap  - (y_overlap_max_boundary - detection_box.y + detection_box.height);

        int overlap_area = max_x_overlap * max_y_overlap;
        float overlap_percentage = overlap_area/detection_box.area();
        if(overlap_percentage > IOUThreshHold)
        {
            // filtered_class_detection_boxes.push_back(detection_box);
            class_detection_boxes.erase(class_detection_boxes.begin() + class_detection_boxes_index);
            class_detection_scores.erase(class_detection_scores.begin() + class_detection_boxes_index);
        }else
        {
            class_detection_boxes_index++;
        }
    }
}

void NonMaxSupression(std::vector<cv::Rect> all_detection_boxes[NUM_CLASSES], std::vector<float> all_detection_scores[NUM_CLASSES], std::vector<cv::Rect> filtered_detection_boxes[NUM_CLASSES], std::vector<float> filtered_detection_scores[NUM_CLASSES])
{
    for(int all_detection_boxes_index = 0; all_detection_boxes_index < NUM_CLASSES; all_detection_boxes_index++)
    {

        std::vector<cv::Rect> class_detection_boxes = all_detection_boxes[all_detection_boxes_index];
        std::vector<float> class_detection_scores = all_detection_scores[all_detection_boxes_index];

            
        while(class_detection_boxes.size() > 0)
        {
            int most_confident_box_index = FindHighestScoringDetectionBox(class_detection_boxes, class_detection_scores);
            cv::Rect most_confident_box = class_detection_boxes[most_confident_box_index];

            filtered_detection_boxes[all_detection_boxes_index].push_back(most_confident_box);
            filtered_detection_scores[all_detection_boxes_index].push_back(class_detection_scores[most_confident_box_index]);
            class_detection_boxes.erase(class_detection_boxes.begin() + most_confident_box_index);
            class_detection_scores.erase(class_detection_scores.begin() + most_confident_box_index);

            IOUFilter(most_confident_box, class_detection_boxes, class_detection_scores, .40, filtered_detection_boxes[all_detection_boxes_index]);
            // //find highest scoring box
            // for(int class_detection_boxes_index = 0; class_detection_boxes_index < class_detection_boxes.size(); class_detection_boxes_index++)
            // {
            //     cv::Rect detection_box = class_detection_boxes[class_detection_boxes_index];
            //     float detection_box_confidence_score = class_detection_scores[class_detection_boxes_index];
            //     if(detection_box_confidence_score > highest_confidence_score)
            //     {
            //         highest_confidence_score = detection_box_confidence_score;
            //         most_confident_box = detection_box;
            //     }
            // }
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
        neural_net.forward(detections, output_names);
        // auto dnn_end = std::chrono::steady_clock::now();

        for (auto &detection : detections)
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
                        detection_boxes[c].push_back(rect);
                        detection_scores[c].push_back(confidence);
                    }
                }
            }
        }
        std::vector<cv::Rect> filtered_detection_boxes[NUM_CLASSES];
        std::vector<float> filtered_detection_scores[NUM_CLASSES];
        // NonMaxSupression(detection_boxes, detection_scores, filtered_detection_boxes, filtered_detection_scores);

        for (int c = 0; c < NUM_CLASSES; c++)
        {
            std::vector<cv::Rect> &class_detection_boxes = detection_boxes[c];
            std::vector<float> class_detection_scores = detection_scores[c];

            for (cv::Rect &class_detection_box: class_detection_boxes)
            {
                const auto color = colors[c % NUM_COLORS];
                cv::rectangle(frame, cv::Point(class_detection_box.x, class_detection_box.y), cv::Point(class_detection_box.x + class_detection_box.width, class_detection_box.y + class_detection_box.height), color, 3);
            }
            class_detection_boxes.clear();
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

