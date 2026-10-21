#pragma once

#include <yaml-cpp/yaml.h>
#include <string>
#include <stdexcept>

namespace backend::tools
{

inline void requireMapping(const YAML::Node& config)
{
    if (!config.IsMap())
        throw std::invalid_argument("Expected a YAML mapping");
}

inline YAML::Node readSection(const YAML::Node& config, const std::string& key,
                              bool required = false)
{
    requireMapping(config);
    const auto node = config[key];
    if (!node)
    {
        if (required)
            throw std::invalid_argument("Missing required YAML section: " + key);
        return YAML::Node(YAML::NodeType::Map);
    }
    if (!node.IsMap())
        throw std::invalid_argument("YAML section must be a mapping: " + key);
    return node;
}

template <typename T>
T readRequiredConfig(const YAML::Node& config, const std::string& key)
{
    requireMapping(config);
    const auto node = config[key];
    if (!node || !node.IsScalar())
        throw std::invalid_argument("Missing or non-scalar YAML value: " + key);
    return node.as<T>();
}

template <typename T>
T readConfig(const YAML::Node& config, const std::string& key, T default_value)
{
    requireMapping(config);
    const auto node = config[key];
    if (!node)
        return default_value;
    if (node.IsNull())
        throw std::invalid_argument("YAML value must not be null: " + key);
    return node.as<T>();
}

} // namespace backend::tools
