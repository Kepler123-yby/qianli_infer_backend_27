#include <yaml-cpp/yaml.h>

namespace backend::tools
{

template <typename T>
T readConfig(const YAML::Node& config, const std::string& key, T default_value)
{
    const auto node = config[key];
    return (node && !node.IsNull()) ? node.as<T>() : default_value;
}

}