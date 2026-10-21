#pragma once

#include "../settings/processing_setting.hpp"

namespace backend::processing
{

class Processor
{
public:
    explicit Processor(const setting::ProcessingConfig& config);
    Processor() = default;

private:
    setting::ProcessingConfig processing_config_;
};

} // namespace backend
