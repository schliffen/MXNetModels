#ifndef VISUALIZE_H
#define VISUALIZE_H

#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <vector>

void visualize(const cv::Mat& image,
               at::Tensor boxes,
               at::Tensor class_ids,
               at::Tensor scores,
               const std::vector<cv::Mat>& masks,
               float score_threshold,
               const std::vector<std::string>& class_names);

#endif  // VISUALIZE_H
