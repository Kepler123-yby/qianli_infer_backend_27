#pragma once

#include <yaml-cpp/yaml.h>
#include <opencv2/opencv.hpp>
#include <string>

#include "extra_tools/yaml_read.cpp"

namespace backend
{

// 推理模式
enum class INFERMODE : int
{
    SYNC = 0,   // 同步
    ASYNC = 1   // 异步
};

// 前后处理模式
enum class PROCESSMODE : int
{   
    NO_WAY      = 0,    // 在前后处理都不使用cuda
    ONLY_PRE    = 1,    // 仅在预处理使用cuda
    ONLY_POST   = 2,    // 仅在后处理使用cuda
    ALL         = 3     // 前后处理都使用cuda
};

// 使用推理后端库
enum class INFERBACKEND : int
{
    TENSORRT = 0,
    OPENVINO = 1
};

struct ModelConfig
{
    std::string model_name;
    std::string model_path;
    INFERMODE mode;
    PROCESSMODE process_mode;
    INFERBACKEND infer_backend;

    ModelConfig(const YAML::Node& config);
    ModelConfig() = default;
    ~ModelConfig() = default;
};

// 目标颜色
enum class COLOR : int
{
    BLUE   = 0,
    RED    = 1,
    WRITE  = 2
};

// 目标类别
enum class ENEMYTYPE : int 
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
    COLOR color;
    ENEMYTYPE type;
    std::array<cv::Point2d, 4> points_2d;
};


class YOLOModel
{
public:
    YOLOModel(const YAML::Node& config) : model_config_(config) {};

    virtual ~YOLOModel() noexcept;

    virtual YOLOResult infer(cv::Mat& img);

    virtual YOLOResult inferAsync(cv::Mat& img);

protected:
    ModelConfig model_config_;
};

class Inference
{
public:
    

private:
    std::unique_ptr<YOLOModel> infer_model_;
};

} // namespace backend