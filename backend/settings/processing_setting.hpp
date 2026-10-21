#pragma once

#include <opencv2/core.hpp>
#include <yaml-cpp/yaml.h>
#include <string>

namespace backend::setting
{

struct PreprocessConfig
{
    // cuda相关配置
    bool use_cuda = false;
    // 1. resize：尺寸待加载模型后确定，不从 YAML 重复配置。
    int width = 0, height = 0;
    // 2. padding
    bool use_center_padding = true;
    cv::Scalar padding_color{124, 124, 124};
    // 3. BGR2RGB、uint8->float32、Normalize、HWC2NCHW
    double normalize_scale = 1.0 / 255.0;

    /**
     * @brief 预处理配置初始化
     *
     * @param config yaml文件配置
     */
    explicit PreprocessConfig(const YAML::Node& config);
    PreprocessConfig() = default;
};

struct PostprocessConfig
{
    std::string decoder = "yolo_v8";
    bool use_cuda = false;
    double confidence_threshold = 0.5;
    double nms_threshold = 0.45; // IoU 阈值，不用于中心距离 NMS。

    explicit PostprocessConfig(const YAML::Node& config);
    PostprocessConfig() = default;
};

struct ProcessingConfig
{
    PreprocessConfig preprocess;
    PostprocessConfig postprocess;

    explicit ProcessingConfig(const YAML::Node& config);
    ProcessingConfig() = default;
};

} // namespace backend::setting
