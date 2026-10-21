#pragma once

#include <memory>
#include <future>

#include "settings/model_setting.hpp"
#include "types/detection_result.hpp"

namespace backend
{

class YOLOModel
{
public:
    explicit YOLOModel(const setting::ModelConfig& config) : model_config_(config) {}

    virtual ~YOLOModel() noexcept;

    virtual types::YOLOResult infer(cv::Mat& img);

    virtual std::future<types::YOLOResult> inferAsync(cv::Mat& img);

protected:
    setting::ModelConfig model_config_;
};

class Inference
{
public:

private:
    std::unique_ptr<YOLOModel> infer_model_;
};

} // namespace backend
