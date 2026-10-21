#include "preprocess_config.hpp"
#include "yaml_utils.hpp"

#include <cmath>
#include <vector>

namespace backend
{
PreprocessConfig::PreprocessConfig(const YAML::Node& config)
{
    const auto implementation = tools::readConfig<std::string>(config, "implementation", "cpu");
    if (implementation != "cpu" && implementation != "cuda")
        throw std::invalid_argument("preprocess.implementation must be cpu or cuda");
    use_cuda = implementation == "cuda";
    use_center_padding = tools::readConfig<bool>(config, "center_padding", use_center_padding);
    normalize_scale = tools::readConfig<double>(config, "normalize_scale", normalize_scale);
    if (!std::isfinite(normalize_scale) || normalize_scale <= 0.0)
        throw std::invalid_argument("preprocess.normalize_scale must be finite and positive");

    const auto color = tools::readConfig<std::vector<double>>(
        config, "padding_color", {124, 124, 124});
    if (color.size() != 3)
        throw std::invalid_argument("preprocess.padding_color must contain 3 channels");
    for (const auto channel : color)
    {
        if (!(channel >= 0.0 && channel <= 255.0))
            throw std::invalid_argument("preprocess.padding_color channels must be in [0, 255]");
    }
    padding_color = cv::Scalar(color[0], color[1], color[2]);
}

} // namespace backend
