#pragma once

#include <yaml-cpp/yaml.h>
#include <string>

namespace backend
{

struct ModelConfig
{
    std::string model_name = "yolov8";
    std::string model_path;

    explicit ModelConfig(const YAML::Node& config);
    ModelConfig() = default;
};

} // namespace backend
