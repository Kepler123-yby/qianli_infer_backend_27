#pragma once

#include "yolo.cpp"

#include <opencv2/dnn.hpp>

#include <algorithm>
#include <cmath>
#include <utility>

namespace backend::postprocess
{

class ArmorV5 : public YOLOBase
{
public:
    ArmorV5(const setting::PostprocessConfig& config, const setting::ResizePadding& rp_config)
    : YOLOBase(config, rp_config)
    {}

private:
    static constexpr int features = 22;
    static constexpr int anchors = 25200;

private:
    /**
     * @brief 解析[1, 25200, 22]，每行依次为四点坐标、objectness logit、4色和9类分数。
     */
    std::vector<types::YOLOResult> parser(const std::vector<float>& output,
                                          std::uint64_t frame_count) override
    {
        if (output.size() != features * anchors)
        {
            tools::logger()->error("[postprocess] 帧 {} ArmorV5输出元素数有误，期望{}，实际{}，跳过本帧",
                                   frame_count, features * anchors, output.size());
            return {};
        }

        Candidates candidates;
        int invalid_count = 0;

        if (!collectCandidates(output, frame_count, candidates, invalid_count))
        {
            tools::logger()->info("[postprocess] 帧 {} 未检测到目标，跳过后续操作", frame_count);
            return {};
        }

        std::vector<int> keep_idxes;
        cv::dnn::NMSBoxes(candidates.boxes, candidates.scores,
                          config_.confidence_threshold, config_.nms_threshold, keep_idxes);

        if (keep_idxes.empty())
        {
            tools::logger()->info("[postprocess] 帧 {} 未检测到目标，跳过后续操作", frame_count);
            return {};
        }

        auto results = buildResults(output, candidates, keep_idxes, frame_count);

        tools::logger()->debug("[postprocess] 帧 {} 检测到{}个目标", frame_count, results.size());

        return results;
    }

    /**
     * @brief 收集通过置信度检查的候选，并计算用于NMS的外接矩形
     * @return 至少有一个有效候选时返回true
     */
    bool collectCandidates(const std::vector<float>& output, std::uint64_t frame_count,
                           Candidates& candidates, int& invalid_count) const
    {
        const double logit_thresh = std::log(config_.confidence_threshold /
                                             (1.0 - config_.confidence_threshold));

        for (int i = 0; i < anchors; ++i)
        {
            const auto base = output.data() + i * features;
            const float obj_logit = base[8];

            if (obj_logit < logit_thresh) continue;

            if (!std::all_of(base, base + features, [](float value) { return std::isfinite(value); }))
            {
                ++invalid_count;
                continue;
            }

            // 统一缩放和平移不改变IoU，因此在模型坐标系完成NMS，仅还原保留结果。
            double min_x = base[0], max_x = base[0];
            double min_y = base[1], max_y = base[1];

            for (int j = 2; j < 8; j += 2)
            {
                min_x = std::min(min_x, static_cast<double>(base[j]));
                max_x = std::max(max_x, static_cast<double>(base[j]));
                min_y = std::min(min_y, static_cast<double>(base[j + 1]));
                max_y = std::max(max_y, static_cast<double>(base[j + 1]));
            }

            if (max_x <= min_x || max_y <= min_y)
            {
                ++invalid_count;
                continue;
            }

            candidates.boxes.emplace_back(min_x, min_y, max_x - min_x, max_y - min_y);
            candidates.scores.push_back(1.0f / (1.0f + std::exp(-obj_logit)));
            candidates.raw_idxes.push_back(i);
        }

        if (invalid_count > 0)
            tools::logger()->warn("[postprocess] 帧 {} ArmorV5跳过{}个非有限值或退化候选",
                                  frame_count, invalid_count);

        return !candidates.boxes.empty();
    }

    /**
     * @brief 对NMS保留的候选分类，并把四点还原到原图坐标
     */
    std::vector<types::YOLOResult> buildResults(const std::vector<float>& output,
                                                const Candidates& candidates,
                                                const std::vector<int>& keep_idxes,
                                                std::uint64_t frame_count) const
    {
        std::vector<types::YOLOResult> results;
        results.reserve(keep_idxes.size());

        for (const auto idx : keep_idxes)
        {
            const auto base = output.data() + candidates.raw_idxes[idx] * features;
            const auto color_score = std::max_element(base + 9, base + 13);
            const auto class_score = std::max_element(base + 13, base + features);

            const auto color = static_cast<types::TargetColor>(color_score - (base + 9));
            const auto type = static_cast<types::TargetType>(class_score - (base + 13));

            std::vector<cv::Point2d> points(4);
            for (int j = 0; j < 4; ++j)
            {
                points[j].x = (base[j * 2] - rp_scale_.padding_left) / rp_scale_.resize_scale;
                points[j].y = (base[j * 2 + 1] - rp_scale_.padding_top) / rp_scale_.resize_scale;
            }

            results.emplace_back(color, type, std::move(points), frame_count);
        }

        return results;
    }
};

} // namespace backend::postprocess
