#include "tensorrt_backend.hpp"
#include "tools/logger.hpp"

#include <exception>
#include <filesystem>
#include <fstream>
#include <limits>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <utility>
#include <type_traits>

namespace backend::inference
{

bool checkCuda(cudaError_t status, const char* call)
{
    if (status == cudaSuccess) return true;

    tools::logger()->error("[tensorrt] {} 失败: {}", call, cudaGetErrorString(status));
    return false;
}

void Logger::log(Severity severity, const char* msg) noexcept
{
    auto level = spdlog::level::debug;
    if (severity == Severity::kINTERNAL_ERROR) level = spdlog::level::critical;
    else if (severity == Severity::kERROR) level = spdlog::level::err;
    else if (severity == Severity::kWARNING) level = spdlog::level::warn;

    tools::logger()->log(level, "[tensorrt] {}", msg);
}

// 读取engine的输入输出名称、元素数和字节数；返回false时已记录具体原因
bool IOInfo::InspectEngine(const nvinfer1::ICudaEngine& engine)
{
    // 本后端按单输入单输出实现，多张量网络在构造阶段就拒绝。
    if (engine.getNbIOTensors() != 2)
    {
        tools::logger()->error("[tensorrt] 当前仅支持单输入单输出的Engine，实际有{}个张量",
                              engine.getNbIOTensors());
        return false;
    }

    input.clear();
    output.clear();
    input_size = {};
    in_elements = out_elements = 0;
    in_bytes = out_bytes = 0;

    // 根据张量方向获取输入输出，不依赖engine中的排列顺序
    for (int i = 0; i < engine.getNbIOTensors(); ++i)
    {
        const char* name = engine.getIOTensorName(i);
        if (engine.getTensorFormat(name) != nvinfer1::TensorFormat::kLINEAR)
        {
            tools::logger()->error("[tensorrt] 张量{}不是线性布局，当前不支持", name);
            return false;
        }

        const auto shape = engine.getTensorShape(name);
        const auto elements = calculateElements(shape);
        const auto bytes = calculateBytes(elements, engine.getTensorDataType(name));

        if (elements == 0 || bytes == 0) return false;

        if (engine.getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT)
        {
            if (shape.nbDims != 4 || shape.d[0] != 1 || shape.d[1] != 3)
            {
                tools::logger()->error("[tensorrt] 模型输入必须为[1, 3, H, W]");
                return false;
            }

            input_size = cv::Size(static_cast<int>(shape.d[3]), static_cast<int>(shape.d[2]));
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
    {
        tools::logger()->error("[tensorrt] Engine必须包含一个输入和一个输出");
        return false;
    }

    return true;
}

size_t IOInfo::calculateElements(const nvinfer1::Dims& shape)
{
    // nbDims小于0表示维度信息无效，不是动态形状。
    if (shape.nbDims < 0)
    {
        tools::logger()->error("[tensorrt] 张量维度无效");
        return 0;
    }

    size_t elements = 1;
    for (int i = 0; i < shape.nbDims; ++i)
    {
        // 动态维度为-1，需要设置context形状后才能确定缓冲区大小
        if (shape.d[i] <= 0)
        {
            tools::logger()->error("[tensorrt] 当前仅支持各维度大于0的静态张量，第{}维为{}", i, shape.d[i]);
            return 0;
        }

        elements *= static_cast<size_t>(shape.d[i]);
    }

    return elements;
}

size_t IOInfo::calculateBytes(size_t elements, nvinfer1::DataType type)
{
    // 字节数按张量实际类型计算，与engine是否启用FP16精度无关。
    switch (type)
    {
    case nvinfer1::DataType::kFLOAT:
        return elements * sizeof(float);

    case nvinfer1::DataType::kHALF:
        return elements * sizeof(std::uint16_t);

    default:
        tools::logger()->error("[tensorrt] 当前仅支持FP32或FP16输入输出");
        return 0;
    }
}

Channel::Channel(nvinfer1::ICudaEngine& engine, const IOInfo& info)
{
    // 每个channel独立持有context，才能让多个worker真正并行执行。
    context.reset(engine.createExecutionContext());
    if (!context)
    {
        tools::logger()->error("[tensorrt] 执行context创建失败");
        return;
    }

    // 每个channel独立使用一条stream，完成事件不参与计时。
    if (!CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking))) return;
    if (!CUDA_CHECK(cudaEventCreateWithFlags(&done, cudaEventDisableTiming))) return;

    // 分配锁页主机缓冲和显存缓冲。
    if (!CUDA_CHECK(cudaMallocHost(&h_in, info.in_bytes))) return;
    if (!CUDA_CHECK(cudaMallocHost(&h_out, info.out_bytes))) return;
    if (!CUDA_CHECK(cudaMalloc(&d_in, info.in_bytes))) return;
    if (!CUDA_CHECK(cudaMalloc(&d_out, info.out_bytes))) return;

    // 绑定当前channel的输入输出地址。
    if (!context->setTensorAddress(info.input.c_str(), d_in) ||
        !context->setTensorAddress(info.output.c_str(), d_out))
    {
        tools::logger()->error("[tensorrt] 输入输出张量地址绑定失败");
        return;
    }

    ready_ = true;
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

    // 兼容只完成部分资源分配时的释放
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
: WorkerPool()
{
    info_ = info;
    capacity_ = std::max<size_t>(capacity, 1);
    device_id_ = deviceId;

    if (workers <= 0)
    {
        tools::logger()->error("[tensorrt] WorkerPool至少需要一个worker");
        return;
    }

    // 引擎要求在建池时检查一次，不与每次提交的数据校验混在一起。
    if (info_.in_bytes != info_.in_elements * sizeof(float) ||
        info_.out_bytes != info_.out_elements * sizeof(float))
    {
        tools::logger()->error("[tensorrt] 当前执行路径仅支持FP32输入输出，FP16需要显式转换");
        return;
    }

    if (!CUDA_CHECK(cudaSetDevice(device_id_))) return;

    // 每个worker独占一个context和stream，避免并发使用同一执行上下文。
    channels_.reserve(static_cast<size_t>(workers));
    for (int i = 0; i < workers; ++i)
    {
        channels_.push_back(std::make_unique<Channel>(engine, info));

        if (!channels_.back()->ready())
        {
            tools::logger()->error("[tensorrt] 第{}个channel初始化失败", i);
            return;
        }
    }

    // 全部channel就绪后再启动线程，构造失败时由析构函数回收已启动的worker。
    threads_.reserve(workers);
    for (int i = 0; i < workers; ++i)
    {
        threads_.emplace_back([this, i] { loop(i); });
    }

    ready_ = true;
}

WorkerPool::~WorkerPool()
{
    shutdown();
}

std::future<std::vector<float>> WorkerPool::submitBorrowed(const Input& input)
{
    if (!ready_)
    {
        tools::logger()->error("[tensorrt] 推理后端未就绪，跳过本次任务");
        return emptyInferenceFuture();
    }

    if (!checkInput(input)) return emptyInferenceFuture();

    auto task = std::make_unique<Task>();
    task->input = input;

    return enqueue(std::move(task));
}

std::future<std::vector<float>> WorkerPool::submitOwned(Input&& input)
{
    if (!ready_)
    {
        tools::logger()->error("[tensorrt] 推理后端未就绪，跳过本次任务");
        return emptyInferenceFuture();
    }

    if (!checkInput(input)) return emptyInferenceFuture();

    auto task = std::make_unique<Task>();
    // GpuMat没有移动构造，显式swap矩阵头，保留原来的接管路径。
    std::visit([&](auto& tensor)
    {
        using Tensor = std::decay_t<decltype(tensor)>;
        if constexpr (std::is_same_v<Tensor, cv::cuda::GpuMat>)
            task->input.emplace<Tensor>().swap(tensor);
        else
            task->input.emplace<Tensor>(std::move(tensor));
    }, input);

    return enqueue(std::move(task));
}

bool WorkerPool::checkInput(const Input& input) const
{
    return std::visit([this](const auto& tensor)
    {
        size_t elements;
        if constexpr (std::is_same_v<std::decay_t<decltype(tensor)>, cv::Mat>)
            elements = tensor.total();
        else
            elements = static_cast<size_t>(tensor.rows) * static_cast<size_t>(tensor.cols);

        if (tensor.type() == CV_32FC1 && tensor.isContinuous() && elements == info_.in_elements)
            return true;

        tools::logger()->error("[tensorrt] 输入应为连续FP32 NCHW数据，期望{}个元素，实际{}个，type={}，continuous={}",
                              info_.in_elements, elements, tensor.type(), tensor.isContinuous());
        return false;
    }, input);
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
        if (thread.joinable()) thread.join();
    }
}

std::future<std::vector<float>> WorkerPool::enqueue(std::unique_ptr<Task> task)
{
    auto future = task->promise.get_future();
    {
        std::unique_lock<std::mutex> lk(mutex_);
        not_full_.wait(lk, [this]
        {
            return close_ || task_queue_.size() < capacity_;
        });

        if (close_)
        {
            tools::logger()->error("[tensorrt] 任务队列已关闭，跳过本次任务");
            task->promise.set_value({});
            return future;
        }

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

    if (!channel)
        tools::logger()->error("[tensorrt] worker {} 无法设置CUDA设备 {}: {}", index, device_id_,
                              cudaGetErrorString(device_status));

    while (true)
    {
        std::unique_ptr<Task> task;
        {
            std::unique_lock<std::mutex> lk(mutex_);
            not_empty_.wait(lk, [this]
            {
                return close_ || !task_queue_.empty();
            });

            // 关闭后不再接收任务，但仍处理完队列中的任务，兑现所有future。
            if (close_ && task_queue_.empty()) return;

            task = std::move(task_queue_.front());
            task_queue_.pop_front();
        }

        not_full_.notify_one();

        if (!channel)
        {
            task->promise.set_value({});
            continue;
        }

        auto output = std::visit([&](const auto& input)
        {
            return runOnChannel(*channel, input);
        }, task->input);

        if (output.empty())
        {
            // 返回空结果前等待已提交操作，避免提前释放输入或复用缓冲。
            cudaStreamSynchronize(channel->stream);
            tools::logger()->error("[tensorrt] worker {} 推理失败，跳过本次任务", index);
        }

        task->promise.set_value(std::move(output));
    }
}

std::vector<float> WorkerPool::runOnChannel(Channel& ch, const cv::Mat& input)
{
    // 恢复CPU输入缓冲，上一帧可能直接绑定了外部显存。
    if (!ch.context->setTensorAddress(info_.input.c_str(), ch.d_in))
    {
        tools::logger()->error("[tensorrt] 输入张量地址绑定失败");
        return {};
    }

    // 输入已在提交时校验，先复制到锁页内存再异步上传。
    std::memcpy(ch.h_in, input.data, info_.in_bytes);

    if (!CUDA_CHECK(cudaMemcpyAsync(ch.d_in, ch.h_in, info_.in_bytes,
                                    cudaMemcpyHostToDevice, ch.stream)))
        return {};

    return inferBound(ch);
}

std::vector<float> WorkerPool::runOnChannel(Channel& ch, const cv::cuda::GpuMat& input)
{
    // 直接绑定输入显存，任务和调用方保证它存活到本次推理完成。
    cudaPointerAttributes attributes{};
    if (!CUDA_CHECK(cudaPointerGetAttributes(&attributes, input.data))) return {};

    if (attributes.device != device_id_)
    {
        tools::logger()->error("[tensorrt] GpuMat必须与WorkerPool位于同一CUDA设备，跳过本帧");
        return {};
    }

    if (!ch.context->setTensorAddress(info_.input.c_str(), input.data))
    {
        tools::logger()->error("[tensorrt] 输入张量地址绑定失败");
        return {};
    }

    return inferBound(ch);
}

std::vector<float> WorkerPool::inferBound(Channel& ch)
{
    // 推理、回拷和完成事件都挂在同一条stream上，顺序由stream本身保证。
    if (!ch.context->enqueueV3(ch.stream))
    {
        tools::logger()->error("[tensorrt] enqueueV3失败");
        return {};
    }

    if (!CUDA_CHECK(cudaMemcpyAsync(ch.h_out, ch.d_out, info_.out_bytes,
                                    cudaMemcpyDeviceToHost, ch.stream)))
        return {};

    if (!CUDA_CHECK(cudaEventRecord(ch.done, ch.stream))) return {};

    // 等待事件而不是整条stream，语义上只同步到本次推理结束。
    if (!CUDA_CHECK(cudaEventSynchronize(ch.done))) return {};

    std::vector<float> output(info_.out_elements);
    std::memcpy(output.data(), ch.h_out, info_.out_bytes);

    return output;
}

TensorRT::TensorRT(const setting::InferenceConfig& config)
: config_(config)
{
    if (config_.worker == 0 || config_.worker > static_cast<size_t>(std::numeric_limits<int>::max()))
    {
        tools::logger()->error("[tensorrt] worker数量必须在[1, INT_MAX]范围内，当前为{}", config_.worker);
        return;
    }

    if (config_.capacity <= 0)
    {
        tools::logger()->error("[tensorrt] 任务队列容量必须大于0，当前为{}", config_.capacity);
        return;
    }

    // 构建、反序列化及channel分配必须使用同一设备。
    if (!CUDA_CHECK(cudaSetDevice(config_.tensorrt.device_id))) return;

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
            if (engine_data.empty()) return;
        }

        else
        {
            // 构建可能耗时几十秒到几分钟，会阻塞构造；缓存命中时只需反序列化。
            tools::logger()->info("[tensorrt] 从ONNX构建engine: {}", model_path.string());

            engine_data = buildEngine(model_path.string(), cache_path.string());
            if (engine_data.empty()) return;

            writeBinaryFile(cache_path.string(), engine_data.data(), engine_data.size());
        }
    }

    else
    {
        try
        {
            engine_data = readEngine(model_path.string());
        }
        catch (const std::exception& e)
        {
            tools::logger()->warn("[tensorrt] Engine读取异常: {}，{}", model_path.string(), e.what());
        }
        // 读取返回空数据或抛异常时，继续到同名ONNX回退路径。
    }

    // 加载执行资源；runtime必须比engine活得更久。
    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    if (!runtime_)
    {
        tools::logger()->error("[tensorrt] runtime创建失败");
        return;
    }

    // TensorRT通常通过空指针报告反序列化失败，不能只依赖异常捕获。
    const auto deserialize = [&]() {
        if (engine_data.empty()) return;
        try
        {
            engine_.reset(runtime_->deserializeCudaEngine(engine_data.data(), engine_data.size()));
        }
        catch (const std::exception& e)
        {
            tools::logger()->warn("[tensorrt] Engine反序列化异常: {}", e.what());
        }
    };
    deserialize();

    if (!engine_ && !is_onnx)
    {
        auto onnx_path = model_path;
        onnx_path.replace_extension(".onnx");
        std::error_code ec;
        if (!std::filesystem::is_regular_file(onnx_path, ec))
        {
            tools::logger()->error("[tensorrt] Engine加载失败，且未找到同名ONNX: {}", onnx_path.string());
            return;
        }

        tools::logger()->warn("[tensorrt] Engine加载失败，从同名ONNX重新构建: {}", onnx_path.string());
        try
        {
            engine_data = buildEngine(onnx_path.string(), model_path.string());
            deserialize();
            // 重建结果加载成功后再替换旧文件。
            if (engine_)
                writeBinaryFile(model_path.string(), engine_data.data(), engine_data.size());
        }
        catch (const std::exception& e)
        {
            tools::logger()->error("[tensorrt] ONNX回退构建异常: {}，{}", onnx_path.string(), e.what());
            return;
        }
    }

    if (!engine_)
    {
        tools::logger()->error("[tensorrt] Engine反序列化失败: {}", model_path.string());
        return;
    }

    IOInfo info;
    if (!info.InspectEngine(*engine_)) return;

    // 工作池持有engine引用，因此engine_先于它创建、后于它销毁。
    worker_pool_ = std::make_unique<WorkerPool>(
        *engine_, info, static_cast<int>(config_.worker),
        static_cast<size_t>(config_.capacity), config_.tensorrt.device_id);

    if (!worker_pool_->ready()) return;

    input_size_ = info.input_size;
    ready_ = true;
    tools::logger()->info("[tensorrt] 后端就绪: device={}，输入{}个元素，输出{}个元素",
                         config_.tensorrt.device_id, info.in_elements, info.out_elements);
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

    if (restore_device) cudaSetDevice(previous_device);
}

std::future<std::vector<float>> TensorRT::submitBorrowed(const Input& input)
{
    if (!ready_)
    {
        tools::logger()->error("[tensorrt] 推理后端未就绪，跳过本次任务");
        return emptyInferenceFuture();
    }

    return worker_pool_->submitBorrowed(input);
}

std::future<std::vector<float>> TensorRT::submitOwned(Input&& input)
{
    if (!ready_)
    {
        tools::logger()->error("[tensorrt] 推理后端未就绪，跳过本次任务");
        return emptyInferenceFuture();
    }

    return worker_pool_->submitOwned(std::move(input));
}

std::vector<char> TensorRT::readEngine(const std::string& engine_path)
{
    // 以ate模式打开，一次拿到文件长度后直接分配，避免逐块读取。
    std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
    if (!file)
    {
        tools::logger()->error("[tensorrt] Engine文件打开失败: {}", engine_path);
        return {};
    }

    const auto length = static_cast<std::streamoff>(file.tellg());
    const auto file_size = static_cast<std::uintmax_t>(length);

    if (file_size > std::numeric_limits<size_t>::max() ||
        file_size > static_cast<std::uintmax_t>(std::numeric_limits<std::streamsize>::max()))
    {
        tools::logger()->error("[tensorrt] Engine文件过大: {}", engine_path);
        return {};
    }

    std::vector<char> data(file_size);

    file.seekg(0, std::ios::beg);
    if (!file.read(data.data(), static_cast<std::streamsize>(length)))
    {
        tools::logger()->error("[tensorrt] Engine数据读取不完整: {}", engine_path);
        return {};
    }

    tools::logger()->info("[tensorrt] Engine文件读取成功: {}，{} bytes", engine_path, data.size());

    return data;
}

std::vector<char> TensorRT::buildEngine(const std::string& onnx_path, const std::string& engine_path)
{
    // 构建期资源在返回前释放，TensorRT对象只保留运行期资源。
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger_));
    if (!builder)
    {
        tools::logger()->error("[tensorrt] builder创建失败");
        return {};
    }

    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0));
    if (!network)
    {
        tools::logger()->error("[tensorrt] network创建失败");
        return {};
    }

    // 传入0表示不启用显式batch维度，batch由网络输入张量自身的第一维表达。
    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger_));
    if (!parser)
    {
        tools::logger()->error("[tensorrt] parser创建失败");
        return {};
    }

    if (!parser->parseFromFile(onnx_path.c_str(),
                               static_cast<int>(nvinfer1::ILogger::Severity::kWARNING)))
    {
        tools::logger()->error("[tensorrt] ONNX模型解析失败: {}", onnx_path);
        return {};
    }

    auto builder_config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!builder_config)
    {
        tools::logger()->error("[tensorrt] builder_config创建失败");
        return {};
    }

    // FP16在构建时启用；反序列化沿用engine中已确定的精度。
    // 输入输出类型保持模型定义，内部允许选择FP16计算。
    builder_config->setFlag(nvinfer1::BuilderFlag::kFP16);
    tools::logger()->info("[tensorrt] 构建精度: 启用FP16（允许FP32回退）");

    auto serialized_engine = std::unique_ptr<nvinfer1::IHostMemory>(
        builder->buildSerializedNetwork(*network, *builder_config));

    if (!serialized_engine)
    {
        tools::logger()->error("[tensorrt] Engine构建失败: {}", onnx_path);
        return {};
    }

    const auto* begin = static_cast<const char*>(serialized_engine->data());
    std::vector<char> data(begin, begin + serialized_engine->size());

    tools::logger()->info("[tensorrt] Engine构建成功，准备保存到: {}，{} bytes", engine_path, data.size());

    return data;
}

void TensorRT::writeBinaryFile(const std::string& path, const void* data, size_t size)
{
    // ofstream的写入长度受streamsize限制，超限直接拒绝而不是截断。
    if (size > static_cast<size_t>(std::numeric_limits<std::streamsize>::max()))
    {
        tools::logger()->error("[tensorrt] Engine文件过大，拒绝写入: {}", path);
        return;
    }

    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file)
    {
        tools::logger()->error("[tensorrt] Engine文件创建失败: {}", path);
        return;
    }

    file.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
    file.close();

    // close之后才能可靠判断写入完整性（磁盘满、配额超限等）。
    if (!file)
    {
        tools::logger()->error("[tensorrt] Engine文件写入失败: {}", path);
        return;
    }

    tools::logger()->info("[tensorrt] Engine文件保存成功: {}", path);
}

} // namespace backend::inference
