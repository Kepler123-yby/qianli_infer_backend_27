#pragma once

#include <opencv2/core.hpp>
#include <yaml-cpp/yaml.h>
#include <string>

namespace backend::setting
{

struct ResizePadding
{
    double resize_scale;
    int padding_left;
    int padding_right;
    int padding_top;
    int padding_bottom;

    /**
     * @brief 计算缩放和填充像素值
     * 
     * @param target_width 目标的宽
     * @param target_height 目标的长
     * @param current_width 当前宽
     * @param current_height 当前长
     * @param use_center_padding 是否居中填充
     */
    void calculateParam(const int target_width, const int target_height,
                        const int current_width, const int current_height,
                        const bool use_center_padding);
};


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
    // 后处理固定在CPU执行。
    std::string decoder = "yolo_v8";
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
