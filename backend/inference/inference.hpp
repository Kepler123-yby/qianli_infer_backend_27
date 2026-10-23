#pragma once

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>

#include <future>
#include <memory>
#include <vector>

#include "backend.hpp"
#include "settings/pipeline_setting.hpp"

namespace backend::inference
{

class Inference
{
public:
    /**
     * @brief 按config.type创建对应的推理后端
     *
     * @param config 推理配置
     * @throws std::invalid_argument 后端类型或队列参数非法
     * @throws std::runtime_error 模型加载、engine构建或执行资源创建失败，
     *         或当前构建未编译所选后端
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
     * @throws std::runtime_error 输入尺寸不符或推理失败
     * @note OpenVINO会把输入复制到请求缓冲；TensorRT的Mat路径需读取输入直到完成，
     *       其GpuMat路径要求输入位于配置设备且生产流已完成。
     */
    std::vector<float> infer(const cv::Mat& input);
    std::vector<float> infer(const cv::cuda::GpuMat& input);

    /**
     * @brief 异步推理，返回future
     *
     * @param input 预处理后的连续、单通道FP32 NCHW数据
     * @throws std::runtime_error 提交阶段错误；执行阶段错误由future.get()抛出
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
