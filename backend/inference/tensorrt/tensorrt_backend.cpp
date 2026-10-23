#include "tensorrt_backend.hpp"

#include <iostream>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <type_traits>
#include <algorithm>

namespace backend::inference
{

/**
 * @brief 检查CUDA调用结果，失败时抛出带调用名和错误串的异常
 *
 * @param status CUDA运行时返回的状态码
 * @param call 被检查的CUDA调用文本，由CUDA_CHECK自动填入
 */
void checkCuda(cudaError_t status, const char* call)
{
    if (status != cudaSuccess)
    {
        throw std::runtime_error(std::string(call) + ": " + cudaGetErrorString(status));
    }
}

void Logger::log(Severity severity, const char* msg) noexcept
{
    // 只输出警告及以上级别，避免TensorRT的INFO把每次推理的日志刷满。
    if (severity <= Severity::kWARNING)
    {
        std::cout << "[TRT]: " << msg << std::endl;
    }
}

void IOInfo::InspectEngine(const nvinfer1::ICudaEngine& engine)
{
    // 本后端按单输入单输出实现，多张量网络在构造阶段就拒绝。
    if (engine.getNbIOTensors() != 2)
        throw std::runtime_error("当前仅支持单输入单输出的Engine！");

    input.clear();
    output.clear();
    in_elements = out_elements = 0;
    in_bytes = out_bytes = 0;

    // 根据张量方向获取输入输出，不依赖engine中的排列顺序
    for (int i = 0; i < engine.getNbIOTensors(); ++i)
    {
        const char* name = engine.getIOTensorName(i);
        if (engine.getTensorFormat(name) != nvinfer1::TensorFormat::kLINEAR)
            throw std::runtime_error("当前仅支持线性布局的输入输出张量！");

        const size_t elements = calculateElements(engine.getTensorShape(name));
        const size_t bytes = calculateBytes(elements, engine.getTensorDataType(name));

        if (engine.getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT)
        {
            input = name;
            in_elements = elements;
            in_bytes = bytes;
        }
        else
        {
            output = name;
            out_elements = elements;
            out_bytes = bytes;
        }
    }

    if (input.empty() || output.empty())
        throw std::runtime_error("Engine 必须包含一个输入和一个输出！");

    // 这里只记录FP16张量的元信息；执行路径目前只支持FP32，WorkerPool构造时会再检查。
}

size_t IOInfo::calculateElements(const nvinfer1::Dims& shape)
{
    // nbDims小于0表示维度信息无效，不是动态形状。
    if (shape.nbDims < 0)
        throw std::runtime_error("张量维度无效！");

    size_t elements = 1;
    for (int i = 0; i < shape.nbDims; ++i)
    {
        // 动态维度为-1，需要设置context形状后才能确定缓冲区大小
        if (shape.d[i] <= 0)
            throw std::runtime_error("当前仅支持各维度大于0的静态张量！");
        elements *= static_cast<size_t>(shape.d[i]);
    }
    return elements;
}

size_t IOInfo::calculateBytes(const size_t elements, const nvinfer1::DataType type)
{
    // 字节数按张量实际类型计算，与engine是否启用FP16精度无关。
    switch (type)
    {
    case nvinfer1::DataType::kFLOAT:
        return elements * sizeof(float);
    case nvinfer1::DataType::kHALF:
        return elements * sizeof(std::uint16_t);
    default:
        throw std::runtime_error("当前仅支持FP32或FP16输入输出！");
    }
}

Channel::Channel(nvinfer1::ICudaEngine& engine, const IOInfo& info)
{
    // 每个channel独立持有context，才能让多个worker真正并行执行。
    context.reset(engine.createExecutionContext());
    if (!context) throw std::runtime_error("context创建失败！");

    try
    {
        // 每个channel独立使用一条stream，完成事件不参与计时
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&done, cudaEventDisableTiming));

        // 分配锁页主机缓冲和显存缓冲
        CUDA_CHECK(cudaMallocHost(&h_in, info.in_bytes));
        CUDA_CHECK(cudaMallocHost(&h_out, info.out_bytes));
        CUDA_CHECK(cudaMalloc(&d_in, info.in_bytes));
        CUDA_CHECK(cudaMalloc(&d_out, info.out_bytes));

        // 绑定当前channel的输入输出地址
        if (!context->setTensorAddress(info.input.c_str(), d_in) ||
            !context->setTensorAddress(info.output.c_str(), d_out))
            throw std::runtime_error("输入输出张量地址绑定失败！");
    }
    catch (...)
    {
        // 构造失败时对象不会执行析构，需要释放已创建的资源
        release();
        throw;
    }
}

Channel::~Channel()
{
    release();
}

void Channel::release() noexcept
{
    // 缓冲区和context必须保留到当前stream上的任务完成
    if (stream) cudaStreamSynchronize(stream);
    context.reset();

    // 析构中和构造失败路径都会调用release，因此每个句柄都要判空。
    if (d_in) cudaFree(d_in);
    if (d_out) cudaFree(d_out);
    if (h_in) cudaFreeHost(h_in);
    if (h_out) cudaFreeHost(h_out);
    if (done) cudaEventDestroy(done);
    if (stream) cudaStreamDestroy(stream);
}

WorkerPool::WorkerPool(nvinfer1::ICudaEngine& engine,
            const IOInfo& info, 
            int workers,
            size_t capacity,
            int deviceId)
: info_(info), capacity_(std::max<size_t>(capacity, 1)), device_id_(deviceId)
{
    if (workers <= 0)
        throw std::invalid_argument("WorkerPool至少需要一个worker！");
    // 引擎要求在建池时检查一次，不与每次提交的数据校验混在一起。
    if (info_.in_bytes != info_.in_elements * sizeof(float) ||
        info_.out_bytes != info_.out_elements * sizeof(float))
        throw std::runtime_error("当前执行路径仅支持FP32输入输出，FP16需要显式转换！");
    CUDA_CHECK(cudaSetDevice(device_id_));

    // 每个worker独占一个context和stream，避免并发使用同一执行上下文。
    // 必须在当前设备上创建，channel内部的显存分配和地址绑定都依赖它。
    channels_.reserve(static_cast<size_t>(workers));
    for (int i = 0; i < workers; ++i)
    {
        channels_.push_back(std::make_unique<Channel>(engine, info));
    }

    try
    {
        // 全部channel就绪后再启动线程，避免worker访问尚未绑定的缓冲区。
        threads_.reserve(workers);
        for (int i = 0; i < workers; ++i)
        {
            threads_.emplace_back([this, i]{ loop(i); });
        }
    }
    catch (...)
    {
        shutdown();   // 已启动的 worker 必须回收
        throw;
    }
}

WorkerPool::~WorkerPool()
{
    shutdown();
}

std::future<std::vector<float>> WorkerPool::submit(cv::Mat input)
{
    // 按值接收：调用方丢弃自己的头部，入队时再深拷贝数据。
    checkInput(input.type(), input.isContinuous(), input.total());
    auto task = std::make_unique<Task>();
    task->owned = std::move(input);
    return enqueue(std::move(task));
}

std::future<std::vector<float>> WorkerPool::submit(cv::cuda::GpuMat input)
{
    checkInput(input.type(), input.isContinuous(),
               static_cast<size_t>(input.cols) * static_cast<size_t>(input.rows));
    auto task = std::make_unique<Task>();
    task->owned = std::move(input);
    return enqueue(std::move(task));
}

std::future<std::vector<float>> WorkerPool::submitBorrowed(const cv::Mat& input)
{
    // 借用路径只拷头部，省掉一次大块内存复制；代价是调用方必须保证数据活到future完成。
    checkInput(input.type(), input.isContinuous(), input.total());
    auto task = std::make_unique<Task>();
    task->owned = input;
    task->borrowed = true;
    return enqueue(std::move(task));
}

std::future<std::vector<float>> WorkerPool::submitBorrowed(const cv::cuda::GpuMat& input)
{
    checkInput(input.type(), input.isContinuous(),
               static_cast<size_t>(input.cols) * static_cast<size_t>(input.rows));
    auto task = std::make_unique<Task>();
    task->owned = input;
    task->borrowed = true;
    return enqueue(std::move(task));
}

void WorkerPool::checkInput(int type, bool continuous, size_t elements) const
{
    if (type != CV_32FC1 || !continuous)
        throw std::runtime_error("输入必须是连续单通道FP32 NCHW数据！");
    if (elements == 0 || elements != info_.in_elements)
        throw std::runtime_error("输入元素数与Engine不符！");
    // 元素数不能证明NCHW排列；布局正确性由预处理调用方保证。
}

void WorkerPool::shutdown()
{
    // 先置位再唤醒：worker看到close后处理完剩余任务才退出，不会丢弃已入队的future。
    {
        std::lock_guard<std::mutex> lk(mutex_);
        close_ = true;
    }

    not_empty_.notify_all();
    not_full_.notify_all();

    for (auto& thread : threads_)
    {
        if (thread.joinable())
        {
            thread.join();
        }
    }
}

Task::Input WorkerPool::cloneInput(const Task::Input& source)
{
    // 普通值拷贝只复制头部；异步任务必须拥有独立的数据快照。
    return std::visit([](const auto& input) -> Task::Input {
        using Input = std::decay_t<decltype(input)>;
        if constexpr (std::is_same_v<Input, cv::cuda::GpuMat>)
        {
            if (input.empty())
                return cv::cuda::GpuMat{};
            // 普通GpuMat::clone可能使用带pitch的分配，显式保留连续布局。
            cv::cuda::GpuMat copy;
            cv::cuda::createContinuous(input.rows, input.cols, input.type(), copy);
            input.copyTo(copy);
            return copy;
        }
        else
            return input.clone();
    }, source);
}

std::future<std::vector<float>> WorkerPool::enqueue(std::unique_ptr<Task> task)
{
    // 在调用线程完成快照，返回后调用方即可复用输入。
    if (!task->borrowed)
        task->owned = cloneInput(task->input());

    auto future = task->promise.get_future();
    {
        std::unique_lock<std::mutex> lk(mutex_);
        not_full_.wait(lk, [this]{
            return close_ || task_queue_.size() < capacity_;
        });

        if (close_) throw std::runtime_error("任务队列已关闭!");
        task_queue_.push_back(std::move(task));
    }

    not_empty_.notify_one();
    return future;
}

void WorkerPool::loop(size_t index)
{
    // cudaSetDevice只影响当前线程，必须在每个worker线程内单独设置一次。
    const auto device_status = cudaSetDevice(device_id_);
    Channel* channel = (device_status == cudaSuccess) ? channels_[index].get() : nullptr;

    while(true)
    {
        std::unique_ptr<Task> task;
        {
            std::unique_lock<std::mutex> lk(mutex_);
            not_empty_.wait(
                lk, [this]{
                    return close_ || !task_queue_.empty();
                }
            );

            // 关闭后不再接收任务，但仍处理完队列中的任务，兑现所有future。
            if (close_ && task_queue_.empty())
                return;

            task = std::move(task_queue_.front());
            task_queue_.pop_front();
        }

        not_full_.notify_one();
        if (!channel)
        {
            task->promise.set_exception(std::make_exception_ptr(
                std::runtime_error("worker 无法设置 CUDA 设备")));
            continue;
        }

        try
        {
            task->promise.set_value(std::visit([&](const auto& input) {
                return runOnChannel(*channel, input);
            }, task->input()));
        }
        catch (...)
        {
            // 即使enqueue或回拷失败，也先等待已提交操作，避免释放输入或复用缓冲。
            const auto error = std::current_exception();
            cudaStreamSynchronize(channel->stream);
            task->promise.set_exception(error);
        }
    }
}

std::vector<float> WorkerPool::runOnChannel(Channel& ch, const cv::Mat& input)
{
    checkInput(input.type(), input.isContinuous(), input.total());

    // 先复制到锁页内存再异步上传：直接传pageable的Mat指针会退化为同步拷贝。
    std::memcpy(ch.h_in, input.data, info_.in_bytes);
    CUDA_CHECK(cudaMemcpyAsync(ch.d_in, ch.h_in, info_.in_bytes,
                               cudaMemcpyHostToDevice, ch.stream));
    return inferBound(ch);
}

std::vector<float> WorkerPool::runOnChannel(Channel& ch, const cv::cuda::GpuMat& input)
{
    checkInput(input.type(), input.isContinuous(),
               static_cast<size_t>(input.rows) * static_cast<size_t>(input.cols));

    // 输入已在显存中，直接做D2D拷贝；跨设备时这里直接拒绝，不做隐式迁移。
    cudaPointerAttributes attributes{};
    CUDA_CHECK(cudaPointerGetAttributes(&attributes, input.data));
    if (attributes.device != device_id_)
        throw std::runtime_error("GpuMat必须与WorkerPool位于同一CUDA设备！");

    CUDA_CHECK(cudaMemcpyAsync(ch.d_in, input.data, info_.in_bytes,
                               cudaMemcpyDeviceToDevice, ch.stream));
    return inferBound(ch);
}

std::vector<float> WorkerPool::inferBound(Channel& ch)
{
    // 推理、回拷和完成事件都挂在同一条stream上，顺序由stream本身保证。
    if (!ch.context->enqueueV3(ch.stream))
        throw std::runtime_error("TensorRT enqueue失败！");

    CUDA_CHECK(cudaMemcpyAsync(ch.h_out, ch.d_out, info_.out_bytes,
                               cudaMemcpyDeviceToHost, ch.stream));
    CUDA_CHECK(cudaEventRecord(ch.done, ch.stream));

    // 等待事件而不是整条stream，语义上只同步到本次推理结束。
    CUDA_CHECK(cudaEventSynchronize(ch.done));

    std::vector<float> output(info_.out_elements);
    std::memcpy(output.data(), ch.h_out, info_.out_bytes);
    return output;
}

TensorRT::TensorRT(const setting::InferenceConfig& config)
: config_(config)
{
    if (config_.worker == 0 || config_.worker > static_cast<size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("worker数量必须在[1, INT_MAX]范围内！");
    if (config_.capacity <= 0)
        throw std::invalid_argument("任务队列容量必须大于0！");
    // 构建、反序列化及channel分配必须使用同一设备。
    CUDA_CHECK(cudaSetDevice(config_.tensorrt.device_id));

    const std::filesystem::path model_path(config_.model_path);
    const std::string extension = model_path.extension().string();
    const bool is_onnx = extension == ".onnx" || extension == ".ONNX";
    std::vector<char> engine_data;

    // ONNX走“优先读缓存、缺失或过期再构建”的路径；其余扩展名一律按已序列化的engine读取。
    if (is_onnx)
    {
        std::filesystem::path cache_path = model_path;
        cache_path.replace_extension(".engine");

        std::error_code ec;

        // 缓存文件名固定为同名.engine；仅当它存在且不早于ONNX时复用，
        // 避免ONNX更新后误用旧engine。
        const bool cache_valid = 
            std::filesystem::is_regular_file(cache_path, ec) &&
            std::filesystem::last_write_time(cache_path, ec) >=
                std::filesystem::last_write_time(model_path, ec);
        
        if (cache_valid)
        {
            engine_data = readEngine(cache_path.string());
        }

        else 
        {
            // 构建可能耗时几十秒到几分钟，会阻塞构造；缓存命中时只需反序列化。
            std::cout << "从 ONNX 构建 TensorRT engine:" << model_path << std::endl;
            engine_data = buildEngine(model_path.string(), cache_path.string());
            writeBinaryFile(cache_path, engine_data.data(), engine_data.size());
        }
    
    } 

    else 
    {
        engine_data = readEngine(model_path.string());
    }

    // 加载执行资源；runtime必须比engine活得更久。
    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    if (!runtime_) throw std::runtime_error("runtime 创建失败！");
    
    engine_.reset(runtime_->deserializeCudaEngine(engine_data.data(), engine_data.size()));
    if (!engine_) throw std::runtime_error("Engine反序列化失败！");

    IOInfo info;
    info.InspectEngine(*engine_);

    // 工作池持有engine引用，因此engine_先于它创建、后于它销毁。
    worker_pool_ = std::make_unique<WorkerPool>(
        *engine_, info, static_cast<int>(config_.worker),
        static_cast<size_t>(config_.capacity), config_.tensorrt.device_id);
}

TensorRT::~TensorRT()
{
    // 析构可能由另一个线程调用：先排空任务，再在资源所属设备释放执行资源。
    int previous_device = 0;
    const bool restore_device = cudaGetDevice(&previous_device) == cudaSuccess;
    cudaSetDevice(config_.tensorrt.device_id);
    worker_pool_.reset();
    engine_.reset();
    runtime_.reset();
    if (restore_device)
        cudaSetDevice(previous_device);
}

std::vector<float> TensorRT::infer(const cv::Mat& input)
{
    // 同步接口走借用路径，省掉一次整帧深拷贝；借用只在本次调用期间有效。
    return worker_pool_->submitBorrowed(input).get();
}

std::vector<float> TensorRT::infer(const cv::cuda::GpuMat& input)
{
    // 借用GpuMat头部，显存内容由worker在channel stream上直接读取。
    return worker_pool_->submitBorrowed(input).get();
}

std::future<std::vector<float>> TensorRT::inferAsync(const cv::Mat& input)
{
    // Mat的clone是纯主机内存复制，不涉及设备切换，可直接交给submit。
    return worker_pool_->submit(input);
}

std::future<std::vector<float>> TensorRT::inferAsync(const cv::cuda::GpuMat& input)
{
    // 深拷贝发生在提交线程，不能依赖worker线程的cudaSetDevice。
    // 这里临时切到目标设备完成GpuMat拷贝，返回前恢复调用方原来的设备。
    int previous_device = 0;
    CUDA_CHECK(cudaGetDevice(&previous_device));
    CUDA_CHECK(cudaSetDevice(config_.tensorrt.device_id));
    try
    {
        cudaPointerAttributes attributes{};
        CUDA_CHECK(cudaPointerGetAttributes(&attributes, input.data));
        if (attributes.device != config_.tensorrt.device_id)
            throw std::invalid_argument("GpuMat必须与TensorRT位于同一CUDA设备！");
        auto result = worker_pool_->submit(input);
        CUDA_CHECK(cudaSetDevice(previous_device));
        return result;
    }
    catch (...)
    {
        cudaSetDevice(previous_device);
        throw;
    }
}

std::vector<char> TensorRT::readEngine(const std::string& engine_path)
{
    // 以ate模式打开，一次拿到文件长度后直接分配，避免逐块读取。
    std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("Engine 文件打开失败: " + engine_path);
    auto end = file.tellg();
    const auto length = static_cast<std::streamoff>(end);
    const auto file_size = static_cast<std::uintmax_t>(length);
    
    if (file_size > std::numeric_limits<size_t>::max() ||
        file_size > static_cast<std::uintmax_t>(std::numeric_limits<std::streamsize>::max()))
        throw std::runtime_error("Engine 文件过大！");
    
    std::vector<char> engine;
    engine.resize(file_size);

    file.seekg(0, std::ios::beg);
    if (!file.read(engine.data(), static_cast<std::streamsize>(length)))
        throw std::runtime_error("Engine 数据读取不完整！");

    std::cout << "Engine 文件读取成功！\n"
                << "Engine 大小: " << engine.size() << " bytes\n";
    return engine;
}

std::vector<char> TensorRT::buildEngine(const std::string& onnx_path, const std::string& engine_path)
{
    // 构建期资源在返回前释放，TensorRT对象只保留运行期资源。
    // engine_path只用于日志；实际写文件由构造函数的writeBinaryFile完成。
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger_));
    if (!builder) throw std::runtime_error("builder创建失败！");

    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0));
    if (!network) throw std::runtime_error("network创建失败！");

    // 传入0表示不启用显式batch维度，batch由网络输入张量自身的第一维表达。

    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger_));
    if (!parser) throw std::runtime_error("parser创建失败！");
    // 解析ONNX模型
    if (!parser->parseFromFile(
        onnx_path.c_str(),
        static_cast<int>(nvinfer1::ILogger::Severity::kWARNING)
    ))
        throw std::runtime_error("ONNX 模型解析失败: " + onnx_path);

    // 创建构建配置
    auto builder_config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!builder_config) throw std::runtime_error("builder_config创建失败！");

    // 构建序列化engine
    auto serialized_engine = std::unique_ptr<nvinfer1::IHostMemory>(
        builder->buildSerializedNetwork(*network, *builder_config));
    if (!serialized_engine) throw std::runtime_error("Engine 构建失败: " + onnx_path);

    const auto* data = static_cast<const char*>(serialized_engine->data());
    std::vector<char> engine_data(data, data + serialized_engine->size());

    std::cout << "Engine 构建成功，准备保存到: " << engine_path << "\n"
              << "Engine 大小: " << engine_data.size() << " bytes\n";
    return engine_data;
}

void TensorRT::writeBinaryFile(const std::string& path, const void *data, const std::size_t size)
{
    // ofstream的写入长度受streamsize限制，超限直接拒绝而不是截断。
    if (size > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max()))
        throw std::runtime_error("Engine 文件过大！");

    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file) throw std::runtime_error("Engine 文件创建失败: " + path);

    file.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
    file.close();

    // close之后才能可靠判断写入完整性（磁盘满、配额超限等）。
    if (!file) throw std::runtime_error("Engine 文件写入失败: " + path);

    std::cout << "Engine 文件保存成功: " << path << std::endl;
}

} // namespace backend::inference
