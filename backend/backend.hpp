#pragma once

#include <memory>

#include "settings/model_config.hpp"
#include "types/detection_result.hpp"

namespace backend
{

class YOLOModel
{
public:
    explicit YOLOModel(const ModelConfig& config) : model_config_(config) {}

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
