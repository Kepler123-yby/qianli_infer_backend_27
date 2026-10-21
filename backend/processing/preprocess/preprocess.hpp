#pragma once

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>

#include <variant>

#include "settings/processing_setting.hpp"

namespace backend::processing
{

using PreprocessOutput = std::variant<cv::Mat, cv::cuda::GpuMat>;

struct ResizePadding
{
    double resize_scale;
    int padding_left;
    int padding_right;
    int padding_top;
    int padding_bottom;

    /**
     * @brief 计算缩放和填充像素值
     * 
     * @param target_width 目标的宽
     * @param target_height 目标的长
     * @param current_width 当前宽
     * @param current_height 当前长
     * @param use_center_padding 是否居中填充
     */
    void calculateParam(const int target_width, const int target_height,
                        const int current_width, const int current_height,
                        const bool use_center_padding);
};

class PreBase
{
public:
    PreBase() = default;
    PreBase(const setting::PreprocessConfig& config) : config_(config) {};

    virtual PreprocessOutput process(const cv::Mat& img) = 0;

    virtual ~PreBase() = default;

protected:
    setting::PreprocessConfig config_;
};

class PreCPU : public PreBase
{
public:
    PreCPU() = default;
    PreCPU(const setting::PreprocessConfig& config);

    PreprocessOutput process(const cv::Mat& img) override;

    ~PreCPU() = default;
private:
    cv::Mat resizeAndPad(const cv::Mat& img, const ResizePadding& params);
    cv::Mat convertAndNormalize(const cv::Mat& img);
    cv::Mat HWC2NCHW(const cv::Mat& img);

    ResizePadding resize_padding_;
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

    ResizePadding resize_padding_;
};

} // namespace processing