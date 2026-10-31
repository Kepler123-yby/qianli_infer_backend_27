#include "preprocess.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/cudaarithm.hpp>
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

cv::Mat PreCPU::resizeAndPad(const cv::Mat& img, const setting::ResizePadding& params)
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

    // 转float32并归一化，拆分通道时完成BGR转RGB。
    cv::cuda::GpuMat float_img = convertAndNormalize(resized_img);

    // HWC转NCHW，只等待预处理流，不同步整个设备。
    auto output = HWC2NCHW(float_img);
    stream_.waitForCompletion();
    return output;
}

cv::cuda::GpuMat PreCUDA::upload(const cv::Mat& img)
{
    // 复用锁页内存，在独立流中异步上传。
    host_img_.create(img.size(), img.type());
    auto host_img = host_img_.createMatHeader();
    img.copyTo(host_img);
    gpu_img_.upload(host_img, stream_);
    return gpu_img_;
}

cv::cuda::GpuMat PreCUDA::resizeAndPad(const cv::cuda::GpuMat& img)
{
    // 等比例缩放
    const int resized_width = config_.width - resize_padding_.padding_left - resize_padding_.padding_right;
    const int resized_height = config_.height - resize_padding_.padding_top - resize_padding_.padding_bottom;
    cv::cuda::resize(img, resized_img_, cv::Size(resized_width, resized_height),
                     0, 0, cv::INTER_LINEAR, stream_);

    if (resize_padding_.padding_top == 0 && resize_padding_.padding_bottom == 0 &&
        resize_padding_.padding_left == 0 && resize_padding_.padding_right == 0)
        return resized_img_;

    // 按配置颜色填充。
    cv::cuda::copyMakeBorder(resized_img_, padded_img_,
                             resize_padding_.padding_top, resize_padding_.padding_bottom,
                             resize_padding_.padding_left, resize_padding_.padding_right,
                             cv::BORDER_CONSTANT, config_.padding_color, stream_);
    return padded_img_;
}

cv::cuda::GpuMat PreCUDA::convertAndNormalize(const cv::cuda::GpuMat& img)
{
    img.convertTo(float_img_, CV_32F, config_.normalize_scale, 0.0, stream_);
    return float_img_;
}

cv::cuda::GpuMat PreCUDA::HWC2NCHW(const cv::cuda::GpuMat& img)
{
    // 仅复用池内独占的输出，仍被推理或调用方持有的帧不会被覆盖。
    auto slot = std::find_if(output_buffers_.begin(), output_buffers_.end(), [](const auto& buffer)
    {
        return buffer.empty() || CV_XADD(buffer.refcount, 0) == 1;
    });
    if (slot == output_buffers_.end())
    {
        output_buffers_.emplace_back();
        slot = output_buffers_.end() - 1;
    }

    // 连续的(3H, W)显存存放RGB平面，交换B、R的输出位置，省去一次cvtColor。
    auto& output = *slot;
    cv::cuda::createContinuous(img.rows * 3, img.cols, CV_32F, output);
    cv::cuda::GpuMat channels[3];
    for (int i = 0; i < 3; ++i)
        channels[i] = output.rowRange((2 - i) * img.rows, (3 - i) * img.rows);

    cv::cuda::split(img, channels, stream_);
    return output;
}

} // namespace backend::processing