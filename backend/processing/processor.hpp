#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "settings/processing_setting.hpp"
#include "types/detection_result.cpp"
#include "preprocess/preprocess.hpp"
#include "processing/postprocess/postprocess.hpp"

namespace backend::processing
{

/**
 * @brief 前后处理入口，推理由Detector持有
 */
class Processor
{
public:
    /**
     * @brief 一帧预处理的结果，缩放参数与张量必须成对使用
     */
    struct PreprocessResult
    {
        PreprocessOutput tensor;
        setting::ResizePadding rp_scale{};
    };

    explicit Processor(const setting::ProcessingConfig& config);

    /**
     * @brief 缩放、填充、归一化并转换为NCHW
     *
     * @param img 原始图像，BGR
     * @return 本帧张量和对应的缩放参数
     */
    PreprocessResult preprocess(const cv::Mat& img);

    /**
     * @brief 解码一帧推理输出
     *
     * @param output 模型原始输出
     * @param rp_scale 该帧预处理返回的缩放参数
     * @param frame_count 输入端分配的帧号
     * @return 检测结果；无目标或解码失败时返回空数组
     */
    std::vector<types::YOLOResult> postprocess(const std::vector<float>& output,
                                               const setting::ResizePadding& rp_scale,
                                               std::uint64_t frame_count);

private:
    setting::ProcessingConfig processing_config_;
    std::unique_ptr<PreBase> preprocess_;
    std::unique_ptr<PostBase> postprocess_;
};

} // namespace backend
