#pragma once

#include <yaml-cpp/yaml.h>
#include <string>

namespace backend::tools
{

template <typename T>
T readConfig(const YAML::Node& config, const std::string& key, T default_value)
{
    if (!config || config.IsNull())
        return default_value;
    const auto node = config[key];
    return (node && !node.IsNull()) ? node.as<T>() : default_value;
}

} // namespace backend::tools
