#include "inference.hpp"
#include "tools/logger.hpp"

#include <string>

// 后端只在头文件可找到时参与编译；否则给出明确的运行时错误，而不是链接失败。
// 注意这里只看头文件是否存在，具体是否链接对应库由构建系统负责。
#if __has_include(<NvInfer.h>)
#include "tensorrt/tensorrt_backend.hpp"
#define BACKEND_HAS_TENSORRT 1
#endif

#if __has_include(<openvino/openvino.hpp>)
#include "openvino/openvino_backend.hpp"
#define BACKEND_HAS_OPENVINO 1
#endif

namespace backend::inference
{

Inference::Inference(const setting::InferenceConfig& config)
: config_(config)
{
    tools::logger()->info("[inference] 开始初始化: model={}，workers={}，capacity={}",
                         config_.model_path, config_.worker, config_.capacity);

    // 只按config.type创建被选中的后端，未选中的后端不加载模型、不占用显存/线程。
    switch (config_.type)
    {
    case setting::InferBackend::TENSORRT:
#ifdef BACKEND_HAS_TENSORRT
        backend_ = std::make_unique<TensorRT>(config_);
        break;
#else
        tools::logger()->error("[inference] 当前构建未编译TensorRT后端，请使用openvino或重新构建");
        return;
#endif

    case setting::InferBackend::OPENVINO:
#ifdef BACKEND_HAS_OPENVINO
        backend_ = std::make_unique<OpenVINO>(config_);
        break;
#else
        tools::logger()->error("[inference] 当前构建未编译OpenVINO后端，请使用tensorrt或重新构建");
        return;
#endif
    }

    if (!backend_)
    {
        tools::logger()->error("[inference] 推理后端创建失败，后续任务将跳过");
        return;
    }

    tools::logger()->info("[inference] 初始化结束: backend={}",
                         config_.type == setting::InferBackend::TENSORRT ? "TensorRT" : "OpenVINO");
}

Inference::~Inference() = default;

Inference::Inference(Inference&&) noexcept = default;

Inference& Inference::operator=(Inference&&) noexcept = default;

std::vector<float> Inference::infer(const cv::Mat& input)
{
    if (!backend_)
    {
        tools::logger()->error("[inference] 推理后端未就绪，跳过本次任务");
        return {};
    }

    return backend_->infer(input);
}

std::vector<float> Inference::infer(const cv::cuda::GpuMat& input)
{
    if (!backend_)
    {
        tools::logger()->error("[inference] 推理后端未就绪，跳过本次任务");
        return {};
    }

    return backend_->infer(input);
}

std::future<std::vector<float>> Inference::inferAsync(const cv::Mat& input)
{
    if (!backend_)
    {
        tools::logger()->error("[inference] 推理后端未就绪，跳过本次任务");
        return emptyInferenceFuture();
    }

    return backend_->inferAsync(input);
}

std::future<std::vector<float>> Inference::inferAsync(const cv::cuda::GpuMat& input)
{
    if (!backend_)
    {
        tools::logger()->error("[inference] 推理后端未就绪，跳过本次任务");
        return emptyInferenceFuture();
    }

    return backend_->inferAsync(input);
}

} // namespace backend::inference
