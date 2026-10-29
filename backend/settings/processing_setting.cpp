#include "processing_setting.hpp"
#include "yaml_utils.cpp"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace backend::setting
{

void ResizePadding::calculateParam(const int target_width, const int target_height,
                                   const int current_width, const int current_height,
                                   const bool use_center_padding)
{
    resize_scale = std::min(static_cast<double>(target_width) / current_width,
                            static_cast<double>(target_height) / current_height);
    const int resized_width = std::max(1, cvRound(current_width * resize_scale));
    const int resized_height = std::max(1, cvRound(current_height * resize_scale));

    padding_left = use_center_padding ? (target_width - resized_width) / 2 : 0;
    padding_right = target_width - resized_width - padding_left;
    padding_top = use_center_padding ? (target_height - resized_height) / 2 : 0;
    padding_bottom = target_height - resized_height - padding_top;
}

/**
 * @brief 预处理相关设置
 * 
 * @param config 预处理配置
 */
PreprocessConfig::PreprocessConfig(const YAML::Node& config)
{
    // 预处理使用设备
    const auto implementation = tools::readConfig<std::string>(config, "implementation", "cpu");
    use_cuda = implementation == "cuda";
    
    // 是否进行中心填充
    use_center_padding = tools::readConfig<bool>(config, "center_padding", use_center_padding);
    
    // 归一化系数
    normalize_scale = tools::readConfig<double>(config, "normalize_scale", normalize_scale);
    
    // 填充颜色
    const auto color = tools::readConfig<std::vector<double>>(
        config, "padding_color", {124, 124, 124});
    padding_color = cv::Scalar(color[0], color[1], color[2]);
}

/**
 * @brief 后处理相关设置
 * 
 * @param config 
 */
PostprocessConfig::PostprocessConfig(const YAML::Node& config)
{
    // 输出解析方式
    decoder = tools::readConfig<std::string>(config, "decoder", decoder);

    // 分类阈值
    confidence_threshold = tools::readConfig<double>(config, "confidence_threshold", confidence_threshold);
    
    // nms阈值
    nms_threshold = tools::readConfig<double>(config, "nms_threshold", nms_threshold);
}

ProcessingConfig::ProcessingConfig(const YAML::Node& config)
    : preprocess(config ? config["preprocess"] : YAML::Node{}),
      postprocess(config ? config["postprocess"] : YAML::Node{})
{
}

} // namespace backend::setting
