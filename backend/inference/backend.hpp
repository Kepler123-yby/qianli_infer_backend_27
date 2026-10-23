#pragma once

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>

#include <future>
#include <vector>

namespace backend::inference
{

/**
 * @brief 与具体后端无关的推理接口
 *
 * @note TensorRT与OpenVINO实现同一组签名；Inference按config.type持有其中一个。
 *       输入统一为预处理后的连续、单通道FP32 NCHW数据，输出为CPU上的FP32数组。
 */
class Backend
{
public:
    virtual ~Backend() = default;

    virtual std::vector<float> infer(const cv::Mat& input) = 0;
    virtual std::vector<float> infer(const cv::cuda::GpuMat& input) = 0;

    virtual std::future<std::vector<float>> inferAsync(const cv::Mat& input) = 0;
    virtual std::future<std::vector<float>> inferAsync(const cv::cuda::GpuMat& input) = 0;
};

} // namespace backend::inference
