#pragma once

#include "yolo.cpp"
#include <opencv2/dnn.hpp>

#include <algorithm>
#include <cmath>
#include <utility>

namespace backend::postprocess
{

class ArmorV8 : public YOLOBase
{
public:
    ArmorV8(const setting::PostprocessConfig& config, const setting::ResizePadding& rp_config)
    : YOLOBase(config, rp_config)
    {}

private:
    static constexpr int features = 21;
    static constexpr int anchors = 6300;

private:
    /**
     * @brief 解析[1, 21, 6300]，每列依次为4色和9类分数、四点坐标。
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
                           Candidates& candidates, int& invalid_count)
    {
        const auto base = output.data();

        for (int i = 0; i < anchors; ++i)
        {
            // 没有objectness分支，检测分数取9类概率的最大值。
            float score = 0;
            for (int j = 4; j < 13; ++j)
            {
                const float value = at(base, j, i);
                if (value > score) score = value;
            }

            if (score < config_.confidence_threshold) continue;

            // 通道13..20为4个关键点，外接矩形仅用于NMS。
            const float first_x = at(base, 13, i);
            const float first_y = at(base, 14, i);
            double min_x = first_x, max_x = first_x;
            double min_y = first_y, max_y = first_y;
            bool valid = std::isfinite(first_x) && std::isfinite(first_y);

            for (int p = 1; p < 4 && valid; ++p)
            {
                const float x = at(base, 13 + p * 2, i);
                const float y = at(base, 14 + p * 2, i);

                valid = std::isfinite(x) && std::isfinite(y);
                if (!valid) break;

                min_x = std::min(min_x, static_cast<double>(x));
                max_x = std::max(max_x, static_cast<double>(x));
                min_y = std::min(min_y, static_cast<double>(y));
                max_y = std::max(max_y, static_cast<double>(y));
            }

            if (!valid || max_x <= min_x || max_y <= min_y)
            {
                ++invalid_count;
                continue;
            }

            candidates.boxes.emplace_back(min_x, min_y, max_x - min_x, max_y - min_y);
            candidates.scores.push_back(score);
            candidates.raw_idxes.push_back(i);
        }

        if (invalid_count > 0)
            tools::logger()->warn("[postprocess] 帧 {} ArmorV8跳过{}个非有限值或退化候选",
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
        const auto base = output.data();

        std::vector<types::YOLOResult> results;
        results.reserve(keep_idxes.size());

        for (const auto idx : keep_idxes)
        {
            const auto anchor = candidates.raw_idxes[idx];

            // 通道4..12为9类概率，取最大值作为目标类型。
            int type = 0;
            float type_score = at(base, 4, anchor);
            for (int j = 5; j < 13; ++j)
            {
                const float value = at(base, j, anchor);
                if (value > type_score)
                {
                    type_score = value;
                    type = j - 4;
                }
            }

            // 通道0..3为4色概率，与类别相互独立。
            int color = 0;
            float color_score = at(base, 0, anchor);
            for (int j = 1; j < 4; ++j)
            {
                const float value = at(base, j, anchor);
                if (value > color_score)
                {
                    color_score = value;
                    color = j;
                }
            }

            // 通道13..20为4个关键点，还原到原图坐标。
            std::vector<cv::Point2d> points(4);
            for (int p = 0; p < 4; ++p)
            {
                const double x = at(base, 13 + p * 2, anchor);
                const double y = at(base, 14 + p * 2, anchor);

                points[p].x = (x - rp_scale_.padding_left) / rp_scale_.resize_scale;
                points[p].y = (y - rp_scale_.padding_top) / rp_scale_.resize_scale;
            }

            results.emplace_back(static_cast<types::TargetColor>(color),
                                 static_cast<types::TargetType>(type),
                                 std::move(points), frame_count);
        }

        return results;
    }

    /**
     * @brief 对当前output做处理，使[1, 6300, 21]按[1, 21, 6300]取值
     */
    float at(const float* output_ptr, int feature, int anchor) const
    {
        return output_ptr[anchors * feature + anchor];
    }

};

} // namespace backend::postprocess
