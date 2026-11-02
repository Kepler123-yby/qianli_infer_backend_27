#pragma once

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <memory>
#include <vector>
#include <future>
#include <deque>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>

#include "inference.hpp"

namespace backend::inference
{

// 返回false时已记录具体CUDA错误，调用方直接跳过即可。
bool checkCuda(cudaError_t status, const char* call);

#define CUDA_CHECK(call) checkCuda((call), #call)

class Logger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char* msg) noexcept override;
};

struct IOInfo
{
    // 输入输出名
    std::string input, output;
    cv::Size input_size;

    // 输入输出元素数和字节数
    size_t in_elements = 0, out_elements = 0;
    size_t in_bytes = 0, out_bytes = 0;

    /**
     * @brief 读取engine的输入输出名称、元素数和字节数，不创建context
     * @return 不支持的网络或格式返回false，具体原因写入日志
     */
    bool InspectEngine(const nvinfer1::ICudaEngine& engine);

private:
    size_t calculateElements(const nvinfer1::Dims& shape);
    size_t calculateBytes(size_t elements, nvinfer1::DataType type);
};

struct Channel
{
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    cudaStream_t stream{};
    cudaEvent_t done{};

    // 锁页主机内存，用于异步上传和下载
    void* h_in = nullptr;
    void* h_out = nullptr;

    // 当前channel独立持有的输入输出显存
    void* d_in = nullptr;
    void* d_out = nullptr;

    /**
     * @brief 创建执行context、stream、完成事件和输入输出缓冲，并绑定张量地址
     * @note 任一步失败时记录日志并保持未就绪，由析构函数回收已分配资源
     */
    Channel(nvinfer1::ICudaEngine& engine, const IOInfo& info);
    Channel() = default;
    ~Channel();

    bool ready() const { return ready_; }

private:
    void release() noexcept;

    bool ready_ = false;
};

struct Task
{
    Input input;                             // 持有CPU/CUDA矩阵引用，不执行数据快照
    std::promise<std::vector<float>> promise;
};

class WorkerPool
{
public:
    WorkerPool(nvinfer1::ICudaEngine& engine,
               const IOInfo& info,
               int workers,
               size_t capacity,
               int deviceId);

    WorkerPool(const WorkerPool&) = delete;
    WorkerPool& operator=(const WorkerPool&) = delete;
    WorkerPool(WorkerPool&&) = delete;
    WorkerPool& operator=(WorkerPool&&) = delete;

    ~WorkerPool();

    bool ready() const { return ready_; }

    // 借用输入，调用方须保持底层内存有效且不修改，直到future完成。
    std::future<std::vector<float>> submitBorrowed(const Input& input);

    // 接管矩阵引用，任务持有到推理完成，不复制数据。
    std::future<std::vector<float>> submitOwned(Input&& input);

private:
    WorkerPool() = default;

    bool checkInput(const Input& input) const;

    void shutdown();

    std::future<std::vector<float>> enqueue(std::unique_ptr<Task> task);

    void loop(size_t index);

    // 输入为已完成预处理的连续FP32 NCHW数据；GpuMat必须在本设备且生产流已完成。
    std::vector<float> runOnChannel(Channel& ch, const cv::Mat& input);
    std::vector<float> runOnChannel(Channel& ch, const cv::cuda::GpuMat& input);
    std::vector<float> inferBound(Channel& ch);

private:
    IOInfo info_;
    size_t capacity_ = 1;
    int device_id_ = 0;
    bool ready_ = false;

    std::vector<std::unique_ptr<Channel>> channels_;
    std::vector<std::thread> threads_;
    std::deque<std::unique_ptr<Task>> task_queue_;
    std::mutex mutex_;
    std::condition_variable not_empty_;
    std::condition_variable not_full_;
    bool close_{false};
};

class TensorRT : public Backend
{
public:
    // 构造时将调用线程的当前CUDA设备设置为config.tensorrt.device_id。
    explicit TensorRT(const setting::InferenceConfig& config);
    ~TensorRT() override;

    cv::Size inputSize() const override { return ready_ ? input_size_ : cv::Size{}; }

    // 输入统一为连续FP32 NCHW数据，CUDA输入须位于配置设备且生产流已完成。
    // CPU输入上传到channel缓冲，CUDA输入直接绑定显存，不在本类执行预处理。
    std::future<std::vector<float>> submitBorrowed(const Input& input) override;
    std::future<std::vector<float>> submitOwned(Input&& input) override;

    TensorRT(const TensorRT&) = delete;
    TensorRT& operator=(const TensorRT&) = delete;
    TensorRT(TensorRT&&) = delete;
    TensorRT& operator=(TensorRT&&) = delete;

private:
    Logger logger_;

    // runtime的生命周期必须覆盖engine
    std::unique_ptr<nvinfer1::IRuntime> runtime_ = nullptr;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_ = nullptr;

    setting::InferenceConfig config_;

    // 声明在engine之后，确保工作池先销毁，再释放engine和runtime。
    std::unique_ptr<WorkerPool> worker_pool_;

    cv::Size input_size_;
    bool ready_ = false;

private:
    std::vector<char> readEngine(const std::string& engine_path);

    /**
     * @brief 解析ONNX模型并构建序列化engine，不创建执行context
     * @return 序列化engine数据；失败时记录日志并返回空数组
     */
    std::vector<char> buildEngine(const std::string& onnx_path, const std::string& engine_path);

    void writeBinaryFile(const std::string& path, const void* data, size_t size);
};

} // namespace backend::inference
