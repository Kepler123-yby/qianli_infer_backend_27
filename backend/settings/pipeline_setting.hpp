#pragma once

#include <string>

#include "processing_setting.hpp"

namespace backend::setting
{

enum class InferBackend : int
{
    TENSORRT = 0,
    OPENVINO = 1
};

enum class InferMode : int
{
    SYNC = 0,
    ASYNC = 1
};

struct InferenceConfig
{
    std::string model_path;
    InferBackend type = InferBackend::TENSORRT;

    struct TensorRTConfig
    {
        int device_id = 0;
    } tensorrt;

    struct OpenVINOConfig
    {
        std::string device = "CPU";
    } openvino;

    InferMode mode = InferMode::SYNC;
    int capacity = 1;
    size_t worker = 1;

    explicit InferenceConfig(const YAML::Node& config);
    InferenceConfig() = default;
};

struct PipelineConfig
{
    ProcessingConfig processing;
    InferenceConfig inference;

    // 接收 pipeline 子节点，不接收完整 YAML 文档。
    explicit PipelineConfig(const YAML::Node& config);
    PipelineConfig() = default;
};

PipelineConfig loadPipelineConfig(const std::string& path);

} // namespace backend::setting
