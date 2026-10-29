#pragma once

#include "processing/processor.hpp"
#include "inference/inference.hpp"
#include "settings/pipeline_setting.hpp"

#include <cstdint>
#include <future>
#include <memory>
#include <vector>

namespace backend
{

class Detector
{
public:
    /**
     * @brief 已提交但尚未取结果的帧
     *
     * 保存推理future、本帧缩放参数和帧号，三者必须一起交给collect使用。
     */
    struct PendingFrame
    {
        std::future<std::vector<float>> output;
        setting::ResizePadding rp_scale{};
        std::uint64_t frame_count = 0;
    };

    explicit Detector(const YAML::Node& config);

    Detector(const Detector&) = delete;
    Detector& operator=(const Detector&) = delete;
    Detector(Detector&&) noexcept;
    Detector &operator=(Detector &&) noexcept;

    ~Detector();

    /**
     * @brief 同步处理一帧：预处理、推理、后处理
     *
     * @param img 原始图像，BGR
     * @param frame_count 输入端分配的帧号
     * @return 检测结果；输入为空、推理失败或无目标时返回空数组
     */
    std::vector<types::YOLOResult> detect(const cv::Mat& img, std::uint64_t frame_count);

    /**
     * @brief 提交一帧，预处理后立即返回，不等待推理完成
     *
     * @param img 原始图像，BGR
     * @param frame_count 输入端分配的帧号
     * @return 待收取的帧；输入为空时output无效，collect会直接跳过
     * @note 推理输入在提交阶段完成深拷贝，返回后可以释放img
     */
    PendingFrame submit(const cv::Mat& img, std::uint64_t frame_count);

    /**
     * @brief 等待指定帧推理完成并执行后处理
     *
     * @param pending submit返回的待收取帧
     * @return 检测结果；future无效、推理失败或无目标时返回空数组
     */
    std::vector<types::YOLOResult> collect(PendingFrame& pending);

private:
    setting::PipelineConfig pipelineConfig_;
    std::shared_ptr<processing::Processor> processor_;
    std::shared_ptr<inference::Inference> inference_;
};

} // namespace backend
