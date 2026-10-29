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
#include <variant>
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
    using Input = std::variant<cv::Mat, cv::cuda::GpuMat>;

    Input owned;                             // 保存Mat/GpuMat头部；非借用任务在enqueue时深拷贝
    bool borrowed = false;                   // 借用任务共享数据，不执行深拷贝
    std::promise<std::vector<float>> promise;

    const Input& input() const { return owned; }
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

    ~WorkerPool();

    bool ready() const { return ready_; }

    // 深拷贝输入后入队；GpuMat须位于池所在设备，调用线程也须在该设备且生产流已完成。
    std::future<std::vector<float>> submit(cv::Mat input);
    std::future<std::vector<float>> submit(cv::cuda::GpuMat input);

    // 仅浅拷贝头部；future完成前不得修改共享数据，外部管理的底层内存必须保持有效。
    std::future<std::vector<float>> submitBorrowed(const cv::Mat& input);
    std::future<std::vector<float>> submitBorrowed(const cv::cuda::GpuMat& input);

private:
    WorkerPool() = default;

    bool checkInput(int type, bool continuous, size_t elements) const;

    void shutdown();

    void cloneInput(Task& task);
    static Task::Input cloneInput(const cv::Mat& input);
    static Task::Input cloneInput(const cv::cuda::GpuMat& input);
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

    // 两种接口均要求预处理后的连续、单通道FP32 NCHW数据，输出为CPU上的FP32数组。
    // GpuMat必须在配置的设备上，且提交前生产流已完成；本类不执行预处理。
    // 同步接口借用输入并等待完成，调用期间不得从其他线程修改输入。
    std::vector<float> infer(const cv::Mat& input) override;
    std::vector<float> infer(const cv::cuda::GpuMat& input) override;

    // 返回前完成深拷贝，之后调用方可复用输入；队列满时提交仍会阻塞。
    // 任一阶段失败都记录日志并返回空结果，不中断下一帧。
    std::future<std::vector<float>> inferAsync(const cv::Mat& input) override;
    std::future<std::vector<float>> inferAsync(const cv::cuda::GpuMat& input) override;

    TensorRT(const TensorRT&) = delete;
    TensorRT& operator=(const TensorRT&) = delete;

private:
    Logger logger_;

    // runtime的生命周期必须覆盖engine
    std::unique_ptr<nvinfer1::IRuntime> runtime_ = nullptr;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_ = nullptr;

    setting::InferenceConfig config_;

    // 声明在engine之后，确保工作池先销毁，再释放engine和runtime。
    std::unique_ptr<WorkerPool> worker_pool_;

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
