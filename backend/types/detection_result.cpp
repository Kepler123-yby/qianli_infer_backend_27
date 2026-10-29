#pragma once

#include <opencv2/core.hpp>

#include <cstdint>
#include <utility>
#include <vector>

namespace backend::types
{

// 目标颜色
enum class TargetColor : int
{
    BLUE   = 0,
    RED    = 1,
    WHITE  = 2,
    PURPLE = 3
};

// 目标类别
enum class TargetType : int
{
    SENTRY       = 0,
    HERO         = 1,
    ENGINEER     = 2,
    INFANTRY_3   = 3,
    INFANTRY_4   = 4,
    INFANTRY_5   = 5,
    OUTPOST      = 6,
    BASE_BOTTOM  = 7,
    BASE_TOP     = 8
};

struct YOLOResult
{
    TargetColor color_{};
    TargetType type_{};
    std::vector<cv::Point2d> points_2d_;
    std::uint64_t frame_count_ = 0; // 输入帧号，同一帧的所有目标共用，不按检测数量计数。

    YOLOResult() = default;
    YOLOResult(TargetColor color, TargetType type, std::vector<cv::Point2d> points_2d,
               std::uint64_t frame_count)
    : color_(color), type_(type), points_2d_(std::move(points_2d)), frame_count_(frame_count)
    {}
};

} // namespace backend::types
