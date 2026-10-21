#pragma once

#include "preprocess_config.hpp"

namespace backend
{

struct PostprocessConfig
{
    bool use_cuda = false;
    double confidence_threshold = 0.5;
    double nms_threshold = 0.45; // IoU 阈值，不用于中心距离 NMS。

    explicit PostprocessConfig(const YAML::Node& config);
};

struct ProcessingConfig
{
    PreprocessConfig preprocess;
    PostprocessConfig postprocess;

    explicit ProcessingConfig(const YAML::Node& config);
};

} // namespace backend
