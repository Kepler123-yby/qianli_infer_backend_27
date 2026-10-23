#include "openvino_backend.hpp"

#include <cstring>
#include <stdexcept>
#include <utility>

namespace backend::inference
{

OpenVINO::OpenVINO(const setting::InferenceConfig& config)
{
    if (config.worker == 0)
        throw std::invalid_argument("OpenVINO至少需要一个worker！");
    if (config.capacity <= 0)
        throw std::invalid_argument("任务队列容量必须大于0！");
    capacity_ = static_cast<size_t>(config.capacity);

    // 只接收已预处理的张量，不在后端隐式执行resize、归一化或布局转换。
    auto model = core_.read_model(config.model_path);
    if (model->inputs().size() != 1 || model->outputs().size() != 1)
        throw std::runtime_error("当前仅支持单输入单输出的OpenVINO模型！");
    if (model->input().get_partial_shape().is_dynamic() ||
        model->output().get_partial_shape().is_dynamic())
        throw std::runtime_error("当前仅支持静态形状的OpenVINO模型！");
    if (model->input().get_element_type() != ov::element::f32 ||
        model->output().get_element_type() != ov::element::f32)
        throw std::runtime_error("当前执行路径仅支持FP32输入输出！");

    compiled_model_ = core_.compile_model(model, config.openvino.device);
    in_elements_ = ov::shape_size(compiled_model_.input().get_shape());
    out_elements_ = ov::shape_size(compiled_model_.output().get_shape());
    if (in_elements_ == 0 || out_elements_ == 0)
        throw std::runtime_error("模型输入输出不能为空！");

    // 先创建全部request，再启动线程，避免worker访问尚未构造的执行资源。
    requests_.reserve(config.worker);
    for (size_t i = 0; i < config.worker; ++i)
        requests_.push_back(compiled_model_.create_infer_request());

    try
    {
        threads_.reserve(config.worker);
        for (size_t i = 0; i < config.worker; ++i)
            threads_.emplace_back([this, i] { workerLoop(i); });
    }
    catch (...)
    {
        // 构造失败时析构函数不会运行，必须回收已启动的线程。
        shutdown();
        throw;
    }
}

OpenVINO::~OpenVINO()
{
    // 排空已入队任务后再销毁request，调用方持有的future仍可获取结果。
    shutdown();
}

std::vector<float> OpenVINO::infer(const cv::Mat& input)
{
    checkInput(input.type(), input.isContinuous(), input.total());
    return enqueue(input).get();
}

std::vector<float> OpenVINO::infer(const cv::cuda::GpuMat& input)
{
    return enqueue(downloadInput(input)).get();
}

std::future<std::vector<float>> OpenVINO::inferAsync(const cv::Mat& input)
{
    checkInput(input.type(), input.isContinuous(), input.total());
    // Mat值拷贝只复制头部；异步任务需要独立的数据快照。
    return enqueue(input.clone());
}

std::future<std::vector<float>> OpenVINO::inferAsync(const cv::cuda::GpuMat& input)
{
    // download已经产生独立的主机缓冲，不再额外clone。
    return enqueue(downloadInput(input));
}

void OpenVINO::checkInput(int type, bool continuous, size_t elements) const
{
    if (type != CV_32FC1 || !continuous)
        throw std::runtime_error("输入必须是连续单通道FP32数据，布局须为NCHW！");
    if (elements != in_elements_)
        throw std::runtime_error("输入元素数与OpenVINO模型不符！");
}

cv::Mat OpenVINO::downloadInput(const cv::cuda::GpuMat& input) const
{
    checkInput(input.type(), input.isContinuous(),
               static_cast<size_t>(input.rows) * static_cast<size_t>(input.cols));
    cv::Mat host_input;
    input.download(host_input);
    return host_input;
}

std::future<std::vector<float>> OpenVINO::enqueue(cv::Mat input)
{
    auto task = std::make_unique<Task>();
    task->input = std::move(input);
    auto future = task->promise.get_future();

    {
        std::unique_lock<std::mutex> lock(mutex_);
        not_full_.wait(lock, [this] {
            return stopping_ || task_queue_.size() < capacity_;
        });
        if (stopping_)
            throw std::runtime_error("任务队列已关闭！");
        task_queue_.push_back(std::move(task));
    }
    not_empty_.notify_one();
    return future;
}

void OpenVINO::workerLoop(size_t index)
{
    auto& request = requests_[index];
    while (true)
    {
        std::unique_ptr<Task> task;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            not_empty_.wait(lock, [this] {
                return stopping_ || !task_queue_.empty();
            });
            // 关闭后继续兑现队列中的future，不丢弃已接受的任务。
            if (stopping_ && task_queue_.empty())
                return;
            task = std::move(task_queue_.front());
            task_queue_.pop_front();
        }
        not_full_.notify_one();

        try
        {
            task->promise.set_value(executeAndCollect(request, task->input));
        }
        catch (...)
        {
            task->promise.set_exception(std::current_exception());
        }
    }
}

std::vector<float> OpenVINO::executeAndCollect(ov::InferRequest& request, const cv::Mat& input)
{
    // 使用request自有输入缓冲，避免request在任务结束后仍引用调用方的数据。
    auto input_tensor = request.get_input_tensor();
    std::memcpy(input_tensor.data<float>(), input.data, in_elements_ * sizeof(float));

    // 异步性由worker队列提供；线程内同步infer，避免再引入callback生命周期管理。
    request.infer();

    auto output_tensor = request.get_output_tensor();
    std::vector<float> output(out_elements_);
    std::memcpy(output.data(), output_tensor.data<const float>(), out_elements_ * sizeof(float));
    return output;
}

void OpenVINO::shutdown()
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
    }
    not_empty_.notify_all();
    not_full_.notify_all();

    for (auto& thread : threads_)
    {
        if (thread.joinable())
            thread.join();
    }
}

} // namespace backend::inference
