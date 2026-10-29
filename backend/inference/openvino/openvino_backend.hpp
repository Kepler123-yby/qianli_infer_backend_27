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

    /**
     * @brief 同步推理CPU输入
     *
     * @param input 连续、单通道FP32 NCHW数据，布局由预处理保证
     * @return 模型输出；输入非法或推理失败时记录日志并返回空数组
     * @note 借用Mat底层数据，调用返回前不得从其他线程修改或释放该内存。
     */
    std::vector<float> infer(const cv::Mat& input) override;

    /**
     * @brief 同步推理CUDA输入
     *
     * @param input 位于OpenVINO设备可见显存中的连续FP32 NCHW数据
     * @return 模型输出；下载或推理失败时记录日志并返回空数组
     * @note 先同步下载到主机内存，不是CUDA与OpenVINO设备之间的零拷贝。
     *       调用线程须处于GpuMat所在CUDA设备，且提交前生产流已完成。
     */
    std::vector<float> infer(const cv::cuda::GpuMat& input) override;

    /**
     * @brief 异步推理CPU输入
     *
     * @param input 连续、单通道FP32 NCHW数据
     * @return 提交成功时返回对应future；未就绪或输入非法时返回已就绪的空结果
     * @note 返回前完成输入深拷贝，之后调用方可复用输入；队列满时提交会阻塞。
     */
    std::future<std::vector<float>> inferAsync(const cv::Mat& input) override;

    /**
     * @brief 异步推理CUDA输入
     *
     * @param input 位于CUDA显存中的连续FP32 NCHW数据
     * @return 提交成功时返回对应future；未就绪、下载或输入非法时返回空结果
     * @note 下载在调用线程同步完成，返回的future只覆盖排队与推理阶段。
     */
    std::future<std::vector<float>> inferAsync(const cv::cuda::GpuMat& input) override;

private:
    OpenVINO() = default;

    /**
     * @brief 单个待推理任务，保存输入快照和结果承诺
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
     * @param input 已完成数据快照的输入
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
