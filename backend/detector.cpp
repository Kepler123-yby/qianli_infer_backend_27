#include "detector.hpp"
#include "tools/logger.hpp"

namespace backend
{

Detector::Detector(const YAML::Node& config)
: pipelineConfig_(config)
{
    processor_ = std::make_shared<processing::Processor>(pipelineConfig_.processing);
    inference_ = std::make_shared<inference::Inference>(pipelineConfig_.inference);
}

std::vector<types::YOLOResult> Detector::detect(const cv::Mat& img, std::uint64_t frame_count)
{
    if (img.empty())
    {
        tools::logger()->warn("[detector] 帧 {} 输入图像为空，跳过本帧", frame_count);
        return {};
    }

    // 1. 预处理，同时拿到本帧的缩放和填充参数。
    auto input = processor_->preprocess(img);

    // 2. 推理，CPU/CUDA张量由variant分派到对应重载。
    auto output = std::visit([this](const auto& tensor)
    {
        return inference_->infer(tensor);
    }, input.tensor);

    if (output.empty())
    {
        tools::logger()->warn("[detector] 帧 {} 推理结果为空，跳过后处理", frame_count);
        return {};
    }

    // 3. 后处理，无目标时由解码器记录日志并返回空结果。
    return processor_->postprocess(output, input.rp_scale, frame_count);
}

Detector::PendingFrame Detector::submit(const cv::Mat& img, std::uint64_t frame_count)
{
    PendingFrame pending;
    pending.frame_count = frame_count;

    if (img.empty())
    {
        tools::logger()->warn("[detector] 帧 {} 输入图像为空，跳过本帧", frame_count);
        return pending;
    }

    // 预处理在调用线程完成，缩放参数随future一起交给collect。
    auto input = processor_->preprocess(img);
    pending.rp_scale = input.rp_scale;

    // 异步接口返回前完成输入深拷贝，之后局部张量可以安全释放。
    pending.output = std::visit([this](const auto& tensor)
    {
        return inference_->inferAsync(tensor);
    }, input.tensor);

    return pending;
}

std::vector<types::YOLOResult> Detector::collect(PendingFrame& pending)
{
    if (!pending.output.valid()) return {};

    auto output = pending.output.get();

    if (output.empty())
    {
        tools::logger()->warn("[detector] 帧 {} 推理结果为空，跳过后处理", pending.frame_count);
        return {};
    }

    return processor_->postprocess(output, pending.rp_scale, pending.frame_count);
}

} // namespace backend
