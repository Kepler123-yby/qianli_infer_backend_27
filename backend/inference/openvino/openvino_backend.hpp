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

/**
 * @brief 基于OpenVINO Runtime的CPU/GPU推理后端
 *
 * @note 只接收预处理完成的张量，不在后端隐式执行resize、归一化或布局转换。
 *       输入统一为连续、单通道FP32 NCHW数据，输出为CPU上的FP32数组。
 *       每个worker独占一个ov::InferRequest，异步能力由任务队列提供。
 */
class OpenVINO : public Backend
{
public:
    /**
     * @brief 加载模型并创建推理请求与worker线程
     *
     * @param config 推理配置，使用config.openvino.device选择设备
     * @note 初始化失败只记录日志，对象保留未就绪状态，后续推理直接返回空结果。
     */
    explicit OpenVINO(const setting::InferenceConfig& config);

    /**
     * @brief 停止接收新任务，排空已入队任务后销毁线程和请求
     */
    ~OpenVINO() override;

    OpenVINO(const OpenVINO&) = delete;
    OpenVINO& operator=(const OpenVINO&) = delete;

    cv::Size inputSize() const override { return ready_ ? input_size_ : cv::Size{}; }

    // 借用CPU/CUDA输入，future完成前调用方须保持底层数据有效且不修改。
    std::future<std::vector<float>> submitBorrowed(const Input& input) override;

    // 接管输入引用，不执行快照。CUDA输入仍需在调用线程下载到主机内存。
    // 调用线程须处于GpuMat所在设备，且提交前生产流已完成。
    std::future<std::vector<float>> submitOwned(Input&& input) override;

private:
    OpenVINO() = default;

    /**
     * @brief 单个待推理任务，持有主机输入和结果承诺
     */
    struct Task
    {
        cv::Mat input;
        std::promise<std::vector<float>> promise;
    };

    /**
     * @brief 读取并编译模型，创建推理请求与worker线程
     *
     * @param config 推理配置
     * @return 初始化成功返回true，失败时已记录具体原因
     */
    bool initialize(const setting::InferenceConfig& config);

    /**
     * @brief 检查输入类型、连续性和元素数是否与模型一致
     *
     * @param type OpenCV矩阵类型，应为CV_32FC1
     * @param continuous 底层数据是否连续
     * @param elements 输入元素总数
     * @return 符合模型输入约定时返回true，否则记录日志并返回false
     */
    bool checkInput(int type, bool continuous, size_t elements) const;

    /**
     * @brief 将GpuMat同步下载为独立的主机内存
     *
     * @param input 显存中的连续FP32数据
     * @return 下载结果；输入非法时返回空Mat
     */
    cv::Mat downloadInput(const cv::cuda::GpuMat& input) const;

    /**
     * @brief 将任务放入队列，等待空闲worker执行
     *
     * @param input 主机输入，矩阵引用保留到任务完成
     * @return 任务结果的future；队列关闭时兑现空结果
     */
    std::future<std::vector<float>> enqueue(cv::Mat input);

    /**
     * @brief worker线程主体，循环取出任务并在独占的InferRequest上执行
     *
     * @param index 线程编号，同时作为requests_下标
     */
    void workerLoop(size_t index);

    /**
     * @brief 在指定InferRequest上完成一次同步推理
     *
     * @param request 当前worker独占的推理请求
     * @param input 连续、单通道FP32 NCHW数据
     * @return 模型输出；异常由worker的标准任务边界收集
     */
    std::vector<float> executeAndCollect(ov::InferRequest& request, const cv::Mat& input);

    /**
     * @brief 置位停止标志，唤醒全部worker并等待其退出
     */
    void shutdown();

private:
    // 每个worker独占一个InferRequest；compiled_model必须比所有request活得更久。
    ov::Core core_;
    ov::CompiledModel compiled_model_;
    std::vector<ov::InferRequest> requests_;

    cv::Size input_size_;        ///< 模型输入宽高
    size_t in_elements_ = 0;      ///< 模型输入元素数
    size_t out_elements_ = 0;     ///< 模型输出元素数
    size_t capacity_ = 1;         ///< 任务队列容量
    bool ready_ = false;          ///< 初始化是否完成，未就绪时推理直接跳过

    std::vector<std::thread> threads_;
    std::deque<std::unique_ptr<Task>> task_queue_;
    std::mutex mutex_;            ///< 保护task_queue_与stopping_
    std::condition_variable not_empty_;   ///< worker等待新任务
    std::condition_variable not_full_;    ///< 提交线程等待队列空位
    bool stopping_ = false;       ///< 置位后不再接收新任务
};

} // namespace backend::inference
