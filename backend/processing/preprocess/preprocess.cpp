#include "preprocess.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>

#include <algorithm>

namespace backend::processing
{

PreCPU::PreCPU(const setting::PreprocessConfig& config)
: PreBase(config)
{
}

PreprocessOutput PreCPU::process(const cv::Mat& img)
{
    // 计算缩放和填充参数
    resize_padding_.calculateParam(config_.width, config_.height, img.cols, img.rows,
                                   config_.use_center_padding);

    // 缩放并填充
    cv::Mat resized_img = resizeAndPad(img, resize_padding_);

    // BGR转RGB，转float32并归一化
    cv::Mat rgb_img = convertAndNormalize(resized_img);

    // HWC转NCHW
    return HWC2NCHW(rgb_img);
}

void ResizePadding::calculateParam(const int target_width, const int target_height,
                                   const int current_width, const int current_height,
                                   const bool use_center_padding)
{
    resize_scale = std::min(static_cast<double>(target_width) / current_width,
                            static_cast<double>(target_height) / current_height);
    const int resized_width = std::max(1, cvRound(current_width * resize_scale));
    const int resized_height = std::max(1, cvRound(current_height * resize_scale));

    padding_left = use_center_padding ? (target_width - resized_width) / 2 : 0;
    padding_right = target_width - resized_width - padding_left;
    padding_top = use_center_padding ? (target_height - resized_height) / 2 : 0;
    padding_bottom = target_height - resized_height - padding_top;
}

cv::Mat PreCPU::resizeAndPad(const cv::Mat& img, const ResizePadding& params)
{
    // 等比例缩放
    const int resized_width = config_.width - params.padding_left - params.padding_right;
    const int resized_height = config_.height - params.padding_top - params.padding_bottom;
    cv::Mat resized_img;
    cv::resize(img, resized_img, cv::Size(resized_width, resized_height), 0, 0, cv::INTER_LINEAR);

    // 按配置颜色填充
    cv::Mat padded_img;
    cv::copyMakeBorder(resized_img, padded_img,
                       params.padding_top, params.padding_bottom,
                       params.padding_left, params.padding_right,
                       cv::BORDER_CONSTANT, config_.padding_color);
    return padded_img;
}

cv::Mat PreCPU::convertAndNormalize(const cv::Mat& img)
{
    cv::Mat rgb_img;
    cv::cvtColor(img, rgb_img, cv::COLOR_BGR2RGB);
    rgb_img.convertTo(rgb_img, CV_32F, config_.normalize_scale);
    return rgb_img;
}

cv::Mat PreCPU::HWC2NCHW(const cv::Mat& img)
{
    const int sizes[] = {1, 3, img.rows, img.cols};
    cv::Mat output(4, sizes, CV_32F);
    cv::Mat channels[3];
    for (int i = 0; i < 3; ++i)
        channels[i] = cv::Mat(img.rows, img.cols, CV_32F, output.ptr<float>(0, i));

    cv::split(img, channels);
    return output;
}

PreCUDA::PreCUDA(const setting::PreprocessConfig& config)
: PreBase(config)
{
}

PreprocessOutput PreCUDA::process(const cv::Mat& img)
{
    // 计算缩放和填充参数
    resize_padding_.calculateParam(config_.width, config_.height, img.cols, img.rows,
                                   config_.use_center_padding);

    // 上传到GPU
    cv::cuda::GpuMat gpu_img = upload(img);

    // 缩放并填充
    cv::cuda::GpuMat resized_img = resizeAndPad(gpu_img);

    // BGR转RGB，转float32并归一化
    cv::cuda::GpuMat rgb_img = convertAndNormalize(resized_img);

    // HWC转NCHW
    return HWC2NCHW(rgb_img);
}

cv::cuda::GpuMat PreCUDA::upload(const cv::Mat& img)
{
    cv::cuda::GpuMat gpu_img;
    gpu_img.upload(img);
    return gpu_img;
}

cv::cuda::GpuMat PreCUDA::resizeAndPad(const cv::cuda::GpuMat& img)
{
    // 等比例缩放
    const int resized_width = config_.width - resize_padding_.padding_left - resize_padding_.padding_right;
    const int resized_height = config_.height - resize_padding_.padding_top - resize_padding_.padding_bottom;
    cv::cuda::GpuMat resized_img;
    cv::cuda::resize(img, resized_img, cv::Size(resized_width, resized_height), 0, 0, cv::INTER_LINEAR);

    // 按配置颜色填充
    cv::cuda::GpuMat padded_img;
    cv::cuda::copyMakeBorder(resized_img, padded_img,
                             resize_padding_.padding_top, resize_padding_.padding_bottom,
                             resize_padding_.padding_left, resize_padding_.padding_right,
                             cv::BORDER_CONSTANT, config_.padding_color);
    return padded_img;
}

cv::cuda::GpuMat PreCUDA::convertAndNormalize(const cv::cuda::GpuMat& img)
{
    cv::cuda::GpuMat rgb_img;
    cv::cuda::cvtColor(img, rgb_img, cv::COLOR_BGR2RGB);

    cv::cuda::GpuMat float_img;
    rgb_img.convertTo(float_img, CV_32F, config_.normalize_scale);
    return float_img;
}

cv::cuda::GpuMat PreCUDA::HWC2NCHW(const cv::cuda::GpuMat& img)
{
    // GpuMat只支持二维，使用连续的(3H, W)单通道显存存放RGB三个平面
    cv::cuda::GpuMat output;
    cv::cuda::createContinuous(img.rows * 3, img.cols, CV_32F, output);
    cv::cuda::GpuMat channels[3];
    for (int i = 0; i < 3; ++i)
        channels[i] = output.rowRange(i * img.rows, (i + 1) * img.rows);

    cv::cuda::split(img, channels);
    return output;
}

} // namespace backend::processing