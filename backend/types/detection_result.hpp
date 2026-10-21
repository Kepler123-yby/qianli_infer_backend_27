#pragma once

#include <opencv2/core.hpp>
#include <array>

namespace backend
{

// 目标颜色
enum class TargetColor : int
{
    BLUE   = 0,
    RED    = 1,
    WHITE  = 2
};

// 目标类别
enum class TargetType : int
{
    SENTRY       = 0,
    HERO         = 1,
    ENGINEER     = 2,
    INFANTRY_3   = 3,
    INFANTRY_4   = 4,
    OUTPOST      = 6,
    BASE_BOTTOM  = 7,
    BASE_TOP     = 8
};

struct YOLOResult
{
    TargetColor color;
    TargetType type;
    std::array<cv::Point2d, 4> points_2d;
};

} // namespace backend
