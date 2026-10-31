#pragma once

#include "yolo.cpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <utility>

namespace backend::postprocess
{

class RuneV8 : public YOLOBase
{
public:
    RuneV8(const setting::PostprocessConfig& config, const setting::ResizePadding& rp_scale)
    : YOLOBase(config, rp_scale)
    {}

private:
    static constexpr int anchors = 6300;
    static constexpr int features = 18;

private:
    /**
     * @brief 解析[1, 18, 6300]，每列依次为3类分数和5组(x, y, 置信度)。
     */
    types::DetectionResults parser(const std::vector<float>& output,
                                   std::uint64_t frame_count) override
    {
        if (output.size() != features * anchors)
        {
            tools::logger()->error("[postprocess] 帧 {} RuneV8输出元素数有误，期望{}，实际{}，跳过本帧",
                                   frame_count, features * anchors, output.size());
            return {};
        }

        const auto base = output.data();
        std::vector<cv::Point2d> keypoints_list;
        std::vector<float> confidence_list;
        std::vector<int> class_ids_list;

        if (!collectData(keypoints_list, confidence_list, class_ids_list, base))
        {
            tools::logger()->info("[postprocess] 帧 {} 未检测到目标，跳过后续操作", frame_count);
            return {};
        }

        std::vector<int> indices;
        centerDistanceNMS(keypoints_list, confidence_list, class_ids_list,
                          config_.nms_threshold, indices);

        if (indices.empty())
        {
            tools::logger()->info("[postprocess] 帧 {} 未检测到目标，跳过后续操作", frame_count);
            return {};
        }

        auto results = buildResults(keypoints_list, class_ids_list, indices);

        tools::logger()->debug("[postprocess] 帧 {} 检测到{}个目标", frame_count, results.size());

        return results;
    }

    /**
     * @brief 收集通过类别置信度检查的候选，并把五点还原到原图坐标
     * @return 至少有一个有效候选时返回true
     */
    bool collectData(std::vector<cv::Point2d>& keypoints_list,
                     std::vector<float>& confidence_list,
                     std::vector<int>& class_ids_list,
                     const float* base) const
    {
        for (int i = 0; i < anchors; ++i)
        {
            int class_id = 0;
            float confidence = at(base, 0, i);

            for (int j = 1; j < 3; ++j)
            {
                const float value = at(base, j, i);
                if (value > confidence)
                {
                    confidence = value;
                    class_id = j;
                }
            }

            if (!std::isfinite(confidence) || confidence < config_.confidence_threshold) continue;

            std::vector<cv::Point2d> keypoints(5);
            bool valid = true;

            for (int kp = 0; kp < 5; ++kp)
            {
                const double x = at(base, 3 + kp * 3, i);
                const double y = at(base, 4 + kp * 3, i);

                valid = std::isfinite(x) && std::isfinite(y);
                if (!valid) break;

                keypoints[kp].x = (x - rp_scale_.padding_left) / rp_scale_.resize_scale;
                keypoints[kp].y = (y - rp_scale_.padding_top) / rp_scale_.resize_scale;
            }

            if (!valid) continue;

            keypoints_list.insert(keypoints_list.end(), keypoints.begin(), keypoints.end());
            confidence_list.push_back(confidence);
            class_ids_list.push_back(class_id);
        }

        return !class_ids_list.empty();
    }

    /**
     * @brief 按原图像素距离执行NMS，同组候选之间相互抑制
     *
     * @param keypoints_list 关键点列表
     * @param confidence_list 置信度列表
     * @param class_ids_list 类别id列表
     * @param center_dist_threshold 距离阈值
     */
    void centerDistanceNMS(
        const std::vector<cv::Point2d>& keypoints_list,
        const std::vector<float>& confidence_list,
        const std::vector<int>& class_ids_list,
        double center_dist_threshold,
        std::vector<int>& indices) const
    {
        indices.clear();

        const int POINTS_PER_DETECTION = 5;
        const int num_detections = confidence_list.size();

        if (keypoints_list.size() != static_cast<std::size_t>(num_detections * POINTS_PER_DETECTION))
        {
            tools::logger()->error("[postprocess] NMS关键点数目错误！");
            return;
        }

        // 计算每个检测的中心点。
        std::vector<cv::Point2d> centers;
        centers.reserve(num_detections);

        for (int i = 0; i < num_detections; ++i)
        {
            const auto kpts = keypoints_list.data() + i * POINTS_PER_DETECTION;

            // 跳过索引2的R点，使用 top、left、right、bottom 四点计算中心。
            cv::Point2d center(0, 0);
            for (int k = 0; k < 5; ++k)
            {
                if (k == 2) continue;
                center.x += kpts[k].x;
                center.y += kpts[k].y;
            }

            center.x /= 4.0;
            center.y /= 4.0;
            centers.push_back(center);
        }

        std::vector<bool> suppressed(num_detections, false);
        const double dist_threshold_sq = center_dist_threshold * center_dist_threshold;

        // 按置信度大小排序。
        std::vector<int> sorted_indices(num_detections);
        std::iota(sorted_indices.begin(), sorted_indices.end(), 0);

        std::sort(sorted_indices.begin(), sorted_indices.end(), [&confidence_list](int a, int b)
        {
            return confidence_list[a] > confidence_list[b];
        });

        auto same_group = [&class_ids_list](int a, int b)
        {
            const int class_a = class_ids_list[a];
            const int class_b = class_ids_list[b];

            return (class_a < 2 && class_b < 2) ||
                   (class_a == 2 && class_b == 2);
        };

        // 同组间按置信度降序抑制，不同组不比较。
        for (std::size_t i = 0; i < sorted_indices.size(); ++i)
        {
            const int idx_i = sorted_indices[i];

            if (suppressed[idx_i]) continue;

            indices.push_back(idx_i);
            const auto& center_i = centers[idx_i];

            // 大符的距离阈值是正常小符/未激活的1/3，平方后除9。
            const double current_dist_threshold_sq = class_ids_list[idx_i] == 2
                ? dist_threshold_sq / 9.0 : dist_threshold_sq;

            for (std::size_t j = i + 1; j < sorted_indices.size(); ++j)
            {
                const int idx_j = sorted_indices[j];

                if (suppressed[idx_j] || !same_group(idx_i, idx_j)) continue;

                const auto& center_j = centers[idx_j];
                const double dx = center_i.x - center_j.x;
                const double dy = center_i.y - center_j.y;
                const double dist_sq = dx * dx + dy * dy;

                if (dist_sq < current_dist_threshold_sq)
                {
                    suppressed[idx_j] = true;
                }
            }
        }
    }

    /**
     * @brief 根据NMS保留的候选构建符文结果
     */
    std::vector<types::RuneResult> buildResults(const std::vector<cv::Point2d>& keypoints_list,
                                                const std::vector<int>& class_ids_list,
                                                const std::vector<int>& indices) const
    {
        std::vector<types::RuneResult> results;
        results.reserve(indices.size());

        for (const auto idx : indices)
        {
            const auto keypoints = keypoints_list.data() + idx * 5;
            types::RuneResult result;
            result.top = keypoints[0];
            result.left = keypoints[1];
            result.point_R = keypoints[2];
            result.right = keypoints[3];
            result.bottom = keypoints[4];
            result.class_id = static_cast<types::RuneType>(class_ids_list[idx]);
            results.emplace_back(std::move(result));
        }

        return results;
    }

    /**
     * @brief 使[1, 18, 6300]的输出结构能够被以[1, 6300, 18]的输出结构的方式被取值
     */
    float at(const float* base, const int feature, const int anchor) const
    {
        return base[feature * anchors + anchor];
    }
};

} // namespace backend::postprocess
