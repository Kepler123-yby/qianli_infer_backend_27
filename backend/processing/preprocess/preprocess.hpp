#pragma once

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>

#include <variant>

#include "settings/processing_setting.hpp"

namespace backend::processing
{

using PreprocessOutput = std::variant<cv::Mat, cv::cuda::GpuMat>;


class PreBase
{
public:
    PreBase() = default;
    PreBase(const setting::PreprocessConfig& config) : config_(config) {};

    virtual PreprocessOutput process(const cv::Mat& img) = 0;

    setting::ResizePadding get_RP() { return resize_padding_; };

    virtual ~PreBase() = default;

protected:
    setting::PreprocessConfig config_;
    setting::ResizePadding resize_padding_;
};

class PreCPU : public PreBase
{
public:
    PreCPU() = default;
    PreCPU(const setting::PreprocessConfig& config);

    PreprocessOutput process(const cv::Mat& img) override;

    ~PreCPU() = default;
private:
    cv::Mat resizeAndPad(const cv::Mat& img, const setting::ResizePadding& params);
    cv::Mat convertAndNormalize(const cv::Mat& img);
    cv::Mat HWC2NCHW(const cv::Mat& img);
};

class PreCUDA : public PreBase
{
public:
    PreCUDA() = default;
    PreCUDA(const setting::PreprocessConfig& config);

    PreprocessOutput process(const cv::Mat& img) override;
    
    ~PreCUDA() = default;
private:
    /**
     * @brief mat转移到GPU
     * 
     * @param img 
     * @return  GpuMat
     */
    cv::cuda::GpuMat upload(const cv::Mat& img);
    
    cv::cuda::GpuMat resizeAndPad(const cv::cuda::GpuMat& img);
    cv::cuda::GpuMat convertAndNormalize(const cv::cuda::GpuMat& img);
    cv::cuda::GpuMat HWC2NCHW(const cv::cuda::GpuMat& img);
};

} // namespace processing