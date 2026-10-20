#include "backend.hpp"

namespace backend
{

template <typename T>
T readConfig(const YAML::Node& config, const std::string& key, T default_value)
{
    const auto node = config[key];
    return (node && !node.IsNull()) ? node.as<T>() : default_value;
}

ModelConfig::ModelConfig(const YAML::Node& config)
{
    // 1. 模型版本名
    model_name = readConfig<std::string>(config, "model_name", "yolov8");
    // 2. model路径
    model_path = readConfig<std::string>(config, "model_path", "asserts/onnx/Infantry-v8n-fp16-20260726-D1.8w-B16.onnx");
    // 3. 推理的方式（同步 or 异步）
    auto NBmode = readConfig<int>(config, "infer_mode", 0);
    mode = static_cast<INFERMODE>(NBmode);
    // 4. 前后处理模式
    auto processNBmode = readConfig<int>(config, "process_mode", 0);
    process_mode = static_cast<PROCESSMODE>(processNBmode);
    // 5. 推理后端
    auto NBbackend = readConfig<int>(config, "infer_backend", 0);
    infer_backend = static_cast<INFERBACKEND>(NBbackend);

}

} // namespace backend