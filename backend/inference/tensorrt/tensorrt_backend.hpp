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

#include "settings/pipeline_setting.hpp"

namespace backend::inference
{

void checkCuda(cudaError_t status, const char* call);

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
    
    // 输入输出元素数
    size_t in_elements = 0, out_elements = 0;

    // 输入输出字节数
    size_t in_bytes = 0, out_bytes = 0;

    /**
     * @brief 读取engine的输入输出名称、元素数和字节数，不创建context
     *
     * @param engine 已反序列化的engine
     * @throws std::runtime_error 非单输入单输出、动态形状或不支持的数据格式
     * @note 当前支持静态形状、线性布局的FP32和FP16张量。
     */
    void InspectEngine(const nvinfer1::ICudaEngine& engine);

private:
    /**
     * @brief 将静态张量各维度相乘，得到元素总数
     */
    size_t calculateElements(const nvinfer1::Dims& shape);

    /**
     * @brief 按张量实际数据类型计算字节数，与engine内部计算精度无关
     */
    size_t calculateBytes(const size_t elements, const nvinfer1::DataType type);
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
     *
     * @param engine channel使用的engine，生命周期必须覆盖channel
     * @param info 通过InspectEngine得到的输入输出信息
     * @throws std::runtime_error context创建、CUDA资源分配或张量绑定失败
     * @note 在engine对应的CUDA设备上创建和销毁channel。
     */
    Channel(nvinfer1::ICudaEngine& engine, const IOInfo& info);

    /**
     * @brief 等待当前stream完成，再释放执行资源和缓冲区
     */
    ~Channel();

private:
    void release() noexcept;
};

struct Task
{
    using Input = std::variant<cv::Mat, cv::cuda::GpuMat>;
    Input owned;                              // 保存Mat/GpuMat头部；非借用任务在enqueue时深拷贝
    bool borrowed = false;                   // 借用任务共享数据，不执行深拷贝
    std::promise<std::vector<float>> promise;
    
    const Input& input() const { return owned; }
};

class WorkerPool
{
public:
    WorkerPool() = delete;
    WorkerPool( nvinfer1::ICudaEngine& engine,
                const IOInfo& info, 
                int workers,
                size_t capacity,
                int deviceId);
                
    
    WorkerPool(const WorkerPool&) = delete;
    WorkerPool& operator = (const WorkerPool&) = delete;

    ~WorkerPool();

    // 深拷贝输入后入队；GpuMat须位于池所在设备，调用线程也须在该设备且生产流已完成。
    std::future<std::vector<float>> submit(cv::Mat input);
    std::future<std::vector<float>> submit(cv::cuda::GpuMat input);

    // 仅浅拷贝头部；future完成前不得修改共享数据，外部管理的底层内存必须保持有效。
    // GpuMat须位于池所在设备，且提交前生产流已完成。
    std::future<std::vector<float>> submitBorrowed(const cv::Mat& input);
    std::future<std::vector<float>> submitBorrowed(const cv::cuda::GpuMat& input);

private:
    void checkInput(int type, bool continuous, size_t elements) const;

    void shutdown();

    Task::Input cloneInput(const Task::Input& input);
    std::future<std::vector<float>> enqueue(std::unique_ptr<Task> task);
    
    void loop(size_t index);

    // 输入为已完成预处理的连续FP32 NCHW数据；GpuMat必须在本设备且生产流已完成。
    std::vector<float> runOnChannel(Channel& ch, const cv::Mat& input);
    std::vector<float> runOnChannel(Channel& ch, const cv::cuda::GpuMat& input);
    std::vector<float> inferBound(Channel& ch);
private:
    IOInfo info_;
    size_t capacity_;
    int device_id_;

    std::vector<std::unique_ptr<Channel>> channels_;
    std::vector<std::thread> threads_;
    std::deque<std::unique_ptr<Task>> task_queue_;
    std::mutex mutex_;
    std::condition_variable not_empty_;
    std::condition_variable not_full_;
    bool close_{false};
};

class TensorRT
{
public:
    // 构造时将调用线程的当前CUDA设备设置为config.tensorrt.device_id。
    explicit TensorRT(const setting::InferenceConfig& config);
    ~TensorRT();

    // 两种接口均要求预处理后的连续、单通道FP32 NCHW数据，输出为CPU上的FP32数组。
    // GpuMat必须在配置的设备上，且提交前生产流已完成；本类不执行预处理。
    // 同步接口借用输入并等待完成，调用期间不得从其他线程修改输入。
    std::vector<float> infer(const cv::Mat& input);
    std::vector<float> infer(const cv::cuda::GpuMat& input);

    // 返回前完成深拷贝，之后调用方可复用输入；队列满时提交仍会阻塞。
    // 提交阶段错误直接抛出，执行阶段错误由future.get()抛出。
    // 接口选择决定同步/异步行为，不读取config.mode；析构会等待已入队任务完成。
    std::future<std::vector<float>> inferAsync(const cv::Mat& input);
    std::future<std::vector<float>> inferAsync(const cv::cuda::GpuMat& input);

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

private:
    std::vector<char> readEngine(const std::string& engine_path);

    /**
     * @brief 解析ONNX模型并构建序列化engine，不创建执行context
     *
     * @param onnx_path ONNX模型路径
     * @param engine_path engine目标路径，用于日志，文件由调用方写入
     * @return 序列化engine数据
     * @throws std::runtime_error TensorRT对象创建、ONNX解析或engine构建失败
     */
    std::vector<char> buildEngine(const std::string& onnx_path, const std::string& engine_path);

    /**
     * @brief 将二进制数据写入文件，覆盖已有内容
     *
     * @param path 输出文件路径
     * @param data 数据起始地址
     * @param size 数据字节数
     * @throws std::runtime_error 数据过大、文件打开或写入失败
     */
    void writeBinaryFile(const std::string& path, const void *data, const std::size_t size);
};

} // namespace backend::inference
