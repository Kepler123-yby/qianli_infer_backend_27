#pragma once

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>

#include <future>
#include <memory>
#include <vector>

#include "settings/pipeline_setting.hpp"

namespace backend::inference
{

// 用于被跳过的任务：仍返回有效且已就绪的future，调用方无需异常分支。
inline std::future<std::vector<float>> emptyInferenceFuture()
{
    std::promise<std::vector<float>> promise;
    auto future = promise.get_future();
    promise.set_value({});
    return future;
}

/**
 * @brief 与具体后端无关的推理接口
 *
 * @note TensorRT与OpenVINO实现同一组签名，由Inference按config.type持有其中一个。
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

class Inference
{
public:
    /**
     * @brief 按config.type创建对应的推理后端
     *
     * @param config 推理配置
     * @note 初始化失败只记录日志，对象保留未就绪状态，后续调用直接返回空结果
     */
    explicit Inference(const setting::InferenceConfig& config);
    ~Inference();

    Inference(const Inference&) = delete;
    Inference& operator=(const Inference&) = delete;
    Inference(Inference&&) noexcept;
    Inference& operator=(Inference&&) noexcept;

    /**
     * @brief 同步推理，返回前等待结果
     *
     * @param input 预处理后的连续、单通道FP32 NCHW数据
     * @return 正常输出；输入校验或worker执行失败时记录error并返回空数组，调用方应跳过该帧
     * @note OpenVINO会把输入复制到请求缓冲；TensorRT的Mat路径需读取输入直到完成，
     *       其GpuMat路径要求输入位于配置设备且生产流已完成。
     */
    std::vector<float> infer(const cv::Mat& input);
    std::vector<float> infer(const cv::cuda::GpuMat& input);

    /**
     * @brief 异步推理，返回future
     *
     * @param input 预处理后的连续、单通道FP32 NCHW数据
     * @return 输入校验失败、队列关闭或worker执行失败时记录error，future.get()返回空数组
     * @note 返回前完成输入复制，之后调用方可复用输入；队列满时提交仍会等待。
     */
    std::future<std::vector<float>> inferAsync(const cv::Mat& input);
    std::future<std::vector<float>> inferAsync(const cv::cuda::GpuMat& input);

    /**
     * @brief 配置中的执行模式，供上层决定调用infer还是inferAsync
     */
    setting::InferMode mode() const { return config_.mode; }

    /**
     * @brief 实际生效的后端类型
     */
    setting::InferBackend type() const { return config_.type; }

private:
    setting::InferenceConfig config_;
    // 指向具体后端；析构定义在源文件，避免上层包含TensorRT/OpenVINO头文件。
    std::unique_ptr<Backend> backend_;
};

} // namespace backend::inference
