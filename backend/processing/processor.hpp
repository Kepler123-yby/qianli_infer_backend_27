#pragma once

#include "../settings/processing_setting.hpp"

namespace backend
{

class Processor
{
public:
    explicit Processor(const setting::ProcessingConfig& config) : processing_config_(config) {}

private:
    setting::ProcessingConfig processing_config_;
};

} // namespace backend
