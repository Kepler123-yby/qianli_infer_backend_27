#include "processing_setting.hpp"
#include "yaml_utils.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace backend::setting
{

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
