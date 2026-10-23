#include "detector.hpp"

namespace backend
{

Detector::Detector(const YAML::Node& config)
: pipelineConfig_(config)
{
    processor_ = std::make_shared<processing::Processor>(pipelineConfig_.processing);
    inference_ = std::make_shared<inference::Inference>(pipelineConfig_.inference);
}

} // namespace backend