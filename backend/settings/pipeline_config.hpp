#pragma once

#include "model_config.hpp"
#include "processing_config.hpp"

namespace backend
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
    InferBackend type;

    struct TensorRTConfig
    {
        int device_id = 0;
    } tensorrt;

    struct OpenVINOConfig
    {
        std::string device = "CPU";
    } openvino;

    explicit InferenceConfig(const YAML::Node& config);
};

struct ExecutionConfig
{
    InferMode mode = InferMode::SYNC;
    int input_capacity = 1;
    int result_capacity = 1;
    std::string overflow = "drop_oldest";

    explicit ExecutionConfig(const YAML::Node& config);
};

struct PipelineConfig
{
    ModelConfig model;
    ProcessingConfig processing;
    InferenceConfig inference;
    ExecutionConfig execution;

    // 接收 pipeline 子节点，不接收完整 YAML 文档。
    explicit PipelineConfig(const YAML::Node& config);
};

PipelineConfig loadPipelineConfig(const std::string& path);

} // namespace backend
