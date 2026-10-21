#include "model_config.hpp"
#include "yaml_utils.hpp"

namespace backend
{

ModelConfig::ModelConfig(const YAML::Node& config)
{
    model_name = tools::readConfig<std::string>(config, "name", model_name);
    model_path = tools::readRequiredConfig<std::string>(config, "path");
    if (model_name.empty() || model_path.empty())
        throw std::invalid_argument("model.name and model.path must not be empty");
}

} // namespace backend
