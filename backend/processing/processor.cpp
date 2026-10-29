#include "processor.hpp"

namespace backend::processing
{

Processor::Processor(const setting::ProcessingConfig& config)
: processing_config_(config)
{
    if (processing_config_.preprocess.use_cuda)
    {
        preprocess_ = std::make_unique<PreCUDA>(processing_config_.preprocess);
    }

    else
    {
        preprocess_ = std::make_unique<PreCPU>(processing_config_.preprocess);
    }

    postprocess_ = std::make_unique<PostBase>(processing_config_.postprocess);
}

Processor::PreprocessResult Processor::preprocess(const cv::Mat& img)
{
    PreprocessResult result;
    result.tensor = preprocess_->process(img);
    result.rp_scale = preprocess_->get_RP();

    return result;
}

std::vector<types::YOLOResult> Processor::postprocess(const std::vector<float>& output,
                                                      const setting::ResizePadding& rp_scale,
                                                      std::uint64_t frame_count)
{
    return postprocess_->process(output, rp_scale, frame_count);
}

} // namespace backend::processing
