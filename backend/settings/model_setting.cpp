#include "model_setting.hpp"
#include "yaml_utils.hpp"

#include <stdexcept>

namespace backend::setting
{

ModelConfig::ModelConfig(const YAML::Node& config)
{
    model_name = tools::readConfig<std::string>(config, "name", model_name);
    model_path = tools::readConfig<std::string>(config, "path", model_path);
    if (model_name.empty() || model_path.empty())
        throw std::invalid_argument("model.name and model.path must not be empty");
}

} // namespace backend
