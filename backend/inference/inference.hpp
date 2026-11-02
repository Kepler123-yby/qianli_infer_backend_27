#pragma once

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>

#include <future>
#include <memory>
#include <variant>
#include <vector>

#include "settings/pipeline_setting.hpp"

namespace backend::inference
{

using Input = std::variant<cv::Mat, cv::cuda::GpuMat>;

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

    // 模型输入宽高；后端未就绪时返回空尺寸。
    virtual cv::Size inputSize() const = 0;

    // 借用输入，调用方须保持数据有效且不修改，直到future完成。
    virtual std::future<std::vector<float>> submitBorrowed(const Input& input) = 0;

    // 接管矩阵引用，不复制数据，任务持有输入直到推理完成。
    virtual std::future<std::vector<float>> submitOwned(Input&& input) = 0;
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
    ~Inference() = default;

    Inference(const Inference&) = delete;
    Inference& operator=(const Inference&) = delete;
    Inference(Inference&&) noexcept = delete;
    Inference& operator=(Inference&&) noexcept = delete;

    /**
     * @brief 模型输入宽高，未就绪时返回空尺寸
     */
    cv::Size inputSize() const;

    /**
     * @brief 统一提交CPU/CUDA输入，始终返回future
     *
     * @param input 预处理后的连续、单通道FP32 NCHW数据
     * @param async 为false时借用输入并等待完成，为true时接管输入后返回
     * @return 同步模式返回已就绪的future；输入非法或推理失败时get()返回空数组
     * @note 异步输入须由矩阵持有底层内存，建议用std::move提交；不再自动深拷贝，
     *       推理完成前不得通过其他别名修改数据。CUDA输入要求位于配置设备且生产流已完成。
     */
    std::future<std::vector<float>> submit(Input input, bool async);

    /**
     * @brief 配置中的执行模式，供上层选择submit的async参数
     */
    setting::InferMode mode() const { return config_.mode; }

    /**
     * @brief 实际生效的后端类型
     */
    setting::InferBackend type() const { return config_.type; }

private:
    setting::InferenceConfig config_;
    // 持有具体后端，避免上层包含TensorRT/OpenVINO头文件。
    std::unique_ptr<Backend> backend_;
};

} // namespace backend::inference
