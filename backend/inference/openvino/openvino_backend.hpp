#pragma once

#include <openvino/openvino.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "inference.hpp"

namespace backend::inference
{

class OpenVINO : public Backend
{
public:
    explicit OpenVINO(const setting::InferenceConfig& config);
    ~OpenVINO() override;

    OpenVINO(const OpenVINO&) = delete;
    OpenVINO& operator=(const OpenVINO&) = delete;

    // 输入为连续、单通道FP32 NCHW数据；布局由预处理保证，输出为CPU上的FP32数组。
    // 同步接口借用Mat数据，调用返回前不得从其他线程修改或释放底层内存。
    std::vector<float> infer(const cv::Mat& input) override;
    std::vector<float> infer(const cv::cuda::GpuMat& input) override;

    // 返回前完成输入深拷贝；队列满时提交会等待，不保证提交过程完全不阻塞。
    // 提交错误直接抛出，执行错误由future.get()抛出；不读取config.mode。
    std::future<std::vector<float>> inferAsync(const cv::Mat& input) override;
    std::future<std::vector<float>> inferAsync(const cv::cuda::GpuMat& input) override;

    // GpuMat重载先同步下载到CPU，不是CUDA与OpenVINO设备之间的零拷贝。
    // 调用线程须处于GpuMat所在CUDA设备，且提交前生产流已完成。

private:
    struct Task
    {
        cv::Mat input;
        std::promise<std::vector<float>> promise;
    };

    void checkInput(int type, bool continuous, size_t elements) const;
    cv::Mat downloadInput(const cv::cuda::GpuMat& input) const;
    std::future<std::vector<float>> enqueue(cv::Mat input);
    void workerLoop(size_t index);
    std::vector<float> executeAndCollect(ov::InferRequest& request, const cv::Mat& input);
    void shutdown();

private:
    // 每个worker独占一个InferRequest；compiled_model必须比所有request活得更久。
    ov::Core core_;
    ov::CompiledModel compiled_model_;
    std::vector<ov::InferRequest> requests_;

    size_t in_elements_ = 0;
    size_t out_elements_ = 0;
    size_t capacity_ = 1;

    std::vector<std::thread> threads_;
    std::deque<std::unique_ptr<Task>> task_queue_;
    std::mutex mutex_;
    std::condition_variable not_empty_;
    std::condition_variable not_full_;
    bool stopping_ = false;
};

} // namespace backend::inference
