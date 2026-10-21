#pragma once

#include "../settings/processing_config.hpp"

namespace backend
{

class Processor
{
public:
    explicit Processor(const ProcessingConfig& config) : processing_config_(config) {}

private:
    ProcessingConfig processing_config_;
};

} // namespace backend
