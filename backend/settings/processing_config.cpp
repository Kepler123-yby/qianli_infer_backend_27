#include "processing_config.hpp"
#include "yaml_utils.hpp"

namespace backend
{

PostprocessConfig::PostprocessConfig(const YAML::Node& config)
{
    const auto implementation = tools::readConfig<std::string>(config, "implementation", "cpu");
    if (implementation != "cpu" && implementation != "cuda")
        throw std::invalid_argument("postprocess.implementation must be cpu or cuda");
    use_cuda = implementation == "cuda";

    confidence_threshold = tools::readConfig<double>(config, "confidence_threshold", confidence_threshold);
    nms_threshold = tools::readConfig<double>(config, "nms_threshold", nms_threshold);
    if (!(confidence_threshold >= 0.0 && confidence_threshold <= 1.0))
        throw std::invalid_argument("postprocess.confidence_threshold must be in [0, 1]");
    if (!(nms_threshold >= 0.0 && nms_threshold <= 1.0))
        throw std::invalid_argument("postprocess.nms_threshold must be in [0, 1]");
}

ProcessingConfig::ProcessingConfig(const YAML::Node& config)
    : preprocess(tools::readSection(config, "preprocess")),
      postprocess(tools::readSection(config, "postprocess"))
{
}

} // namespace backend
