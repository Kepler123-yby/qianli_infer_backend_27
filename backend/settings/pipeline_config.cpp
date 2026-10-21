#include "pipeline_config.hpp"
#include "yaml_utils.hpp"

namespace backend
{

InferenceConfig::InferenceConfig(const YAML::Node& config)
{
    const auto backend = tools::readRequiredConfig<std::string>(config, "type");
    if (backend == "tensorrt")
    {
        type = InferBackend::TENSORRT;
        const auto node = tools::readSection(config, "tensorrt");
        tensorrt.device_id = tools::readConfig<int>(node, "device_id", tensorrt.device_id);
        if (tensorrt.device_id < 0)
            throw std::invalid_argument("inference.tensorrt.device_id must be non-negative");
    }
    else if (backend == "openvino")
    {
        type = InferBackend::OPENVINO;
        const auto node = tools::readSection(config, "openvino");
        openvino.device = tools::readConfig<std::string>(node, "device", openvino.device);
        if (openvino.device.empty())
            throw std::invalid_argument("inference.openvino.device must not be empty");
    }
    else
    {
        throw std::invalid_argument("inference.type must be tensorrt or openvino");
    }
}

ExecutionConfig::ExecutionConfig(const YAML::Node& config)
{
    const auto value = tools::readConfig<std::string>(config, "mode", "sync");
    if (value == "sync")
        mode = InferMode::SYNC;
    else if (value == "async")
        mode = InferMode::ASYNC;
    else
        throw std::invalid_argument("execution.mode must be sync or async");

    input_capacity = tools::readConfig<int>(config, "input_capacity", input_capacity);
    result_capacity = tools::readConfig<int>(config, "result_capacity", result_capacity);
    overflow = tools::readConfig<std::string>(config, "overflow", overflow);
    if (input_capacity < 1 || result_capacity < 1)
        throw std::invalid_argument("execution queue capacities must be positive");
    if (overflow != "drop_oldest")
        throw std::invalid_argument("execution.overflow must be drop_oldest");
}

PipelineConfig::PipelineConfig(const YAML::Node& config)
    : model(tools::readSection(config, "model", true)),
      processing(tools::readSection(config, "processing")),
      inference(tools::readSection(config, "inference", true)),
      execution(tools::readSection(config, "execution"))
{
}

PipelineConfig loadPipelineConfig(const std::string& path)
{
    const auto root = YAML::LoadFile(path);
    return PipelineConfig(tools::readSection(root, "pipeline", true));
}

} // namespace backend
