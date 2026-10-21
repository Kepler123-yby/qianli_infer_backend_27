#pragma once

#include "settings/pipeline_setting.hpp"

namespace backend::inference
{

class Inference 
{
public:
    Inference(const setting::InferenceConfig& config);
    Inference() = default;

private:
    
};

} // namespace backend::inference