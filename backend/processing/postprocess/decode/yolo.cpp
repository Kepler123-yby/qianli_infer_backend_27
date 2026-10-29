#pragma once

#include "types/detection_result.cpp"
#include "settings/processing_setting.hpp"
#include "tools/logger.hpp"

#include <opencv2/core.hpp>

#include <cmath>
#include <cstdint>
#include <vector>

namespace backend::postprocess
{

struct Candidates
{
    std::vector<cv::Rect2d> boxes;
    std::vector<float> scores;
    std::vector<int> raw_idxes; // 紧凑候选数组对应的原始anchor索引。
};

class YOLOBase
{
public:
    YOLOBase() = default;
    YOLOBase(const setting::PostprocessConfig& config, const setting::ResizePadding& rp_scale)
    : config_(config), rp_scale_(rp_scale)
    {}

    virtual ~YOLOBase() = default;

    /**
     * @brief 解码一帧；无目标或单帧失败时记录日志并返回空结果，调用方应跳过后续操作。
     * @param output 只读的推理输出
     * @param frame_count 输入端分配的帧号，异步完成顺序不影响该编号
     */
    std::vector<types::YOLOResult> postprocess(const std::vector<float>& output,
                                               std::uint64_t frame_count)
    {
        if (output.empty())
        {
            tools::logger()->warn("[postprocess] 帧 {} 推理输出为空，跳过后处理", frame_count);
            return {};
        }

        if (!(config_.confidence_threshold > 0.0 && config_.confidence_threshold < 1.0 &&
              config_.nms_threshold >= 0.0 && config_.nms_threshold <= 1.0 &&
              rp_scale_.resize_scale > 0.0 && std::isfinite(rp_scale_.resize_scale)))
        {
            tools::logger()->error("[postprocess] 帧 {} 参数无效，confidence={}，nms={}，scale={}",
                                   frame_count, config_.confidence_threshold,
                                   config_.nms_threshold, rp_scale_.resize_scale);
            return {};
        }

        return parser(output, frame_count);
    }

protected:
    setting::PostprocessConfig config_;
    setting::ResizePadding rp_scale_{};

    virtual std::vector<types::YOLOResult> parser(const std::vector<float>& output,
                                                  std::uint64_t frame_count) = 0;
};

} // namespace backend::postprocess
