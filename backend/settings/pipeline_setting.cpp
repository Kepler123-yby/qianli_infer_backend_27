#include "pipeline_setting.hpp"
#include "yaml_utils.hpp"

#include <stdexcept>

namespace backend::setting
{

InferenceConfig::InferenceConfig(const YAML::Node& config)
{
    // 模型路径
    model_path = tools::readConfig<std::string>(config, "model_path", model_path);

    // 推理后端
    const auto backend = tools::readConfig<std::string>(config, "type", "");
    if (backend == "tensorrt")
    {
        type = InferBackend::TENSORRT;
        const auto node = config["tensorrt"];
        tensorrt.device_id = tools::readConfig<int>(node, "device_id", tensorrt.device_id);
    }
    else if (backend == "openvino")
    {
        type = InferBackend::OPENVINO;
        const auto node = config["openvino"];
        openvino.device = tools::readConfig<std::string>(node, "device", openvino.device);
    }
    else
    {
        throw std::invalid_argument("推理后端必须是tensorrt或者openvino之一");
    }

    // 同步或异步
    const auto value = tools::readConfig<std::string>(config, "mode", "sync");
    if (value == "sync")
        mode = InferMode::SYNC;
    else if (value == "async")
        mode = InferMode::ASYNC;
    else
        throw std::invalid_argument("执行方式必须为同步或异步中的一种");

    // 队列配置
    capacity = tools::readConfig<size_t>(config, "capacity", capacity);
    worker = tools::readConfig<int>(config, "worker", worker);
}

PipelineConfig::PipelineConfig(const YAML::Node& config)
    : processing(config["processing"]),
      inference(config["inference"])
{
}

PipelineConfig loadPipelineConfig(const std::string& path)
{
    const auto root = YAML::LoadFile(path);
    return PipelineConfig(root["pipeline"]);
}

} // namespace backend::setting
