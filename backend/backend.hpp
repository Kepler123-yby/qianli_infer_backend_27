#pragma once

#include "processing/processor.hpp"
#include "inference/inference.hpp"
#include "settings/pipeline_setting.hpp"

#include <memory>

namespace backend
{

class Detector
{
public:
    explicit Detector(const YAML::Node& config);

private:
    setting::PipelineConfig pipelineConfig_;
    std::shared_ptr<processing::Processor> processor_;
    std::shared_ptr<inference::Inference> inference_;
};

} // namespace backend
