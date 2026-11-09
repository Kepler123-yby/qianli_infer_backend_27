#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <optional>
#include <queue>
#include <string>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

#include <opencv2/core/mat.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <yaml-cpp/node/parse.h>

#include "detector.hpp"
#include "tools/logger.hpp"
#include "types/detection_result.cpp"

const std::string keys =
    "{help h usage ?  |                        | 输出命令行参数说明           }"
    "{config-path  c  | configs/test_demo.yaml | yaml配置文件的路径          }"
    "{start_index  s  | 0                      | 测试视频的开始帧            }"
    "{end_index    e  | 0                      | 结束帧（含），0表示视频末尾  }"
    "{queue_size   qs | 8                      | 队列长度                   }"
    "{display      d  | 0                      | 是否可视化,0:false; 1:true }"
    "{print_video_fps | false                  | 打印解码加delay的平均输出FPS }"
    "{@input-path     | assets/video/装甲板.mp4  | 测试视频路径                }";

double calculate_cost(
    const std::function<backend::types::DetectionResults()>& f,
    backend::types::DetectionResults& result)
{
    auto start = std::chrono::steady_clock::now();
    result = f();
    auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(end - start).count();
}


void draw_box(cv::Mat& img, const std::vector<cv::Point>& points,
              const std::string& label)
{
    if (points.empty()) return;
    const auto box = cv::boundingRect(points);
    const cv::Scalar color(0, 255, 0);
    cv::rectangle(img, box, color, 2, cv::LINE_AA);
    cv::putText(img, label, cv::Point(std::max(0, box.x), std::max(18, box.y - 6)),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv::LINE_AA);
}

void draw_detection(cv::Mat& img, const backend::types::YOLOResult& result)
{
    const char* label = "Unknown";
    using backend::types::TargetType;
    switch (result.type_)
    {
        case TargetType::SENTRY: label = "Sentry"; break;
        case TargetType::HERO: label = "Hero"; break;
        case TargetType::ENGINEER: label = "Engineer"; break;
        case TargetType::INFANTRY_3: label = "Infantry 3"; break;
        case TargetType::INFANTRY_4: label = "Infantry 4"; break;
        case TargetType::INFANTRY_5: label = "Infantry 5"; break;
        case TargetType::OUTPOST: label = "Outpost"; break;
        case TargetType::BASE_BOTTOM: label = "Base bottom"; break;
        case TargetType::BASE_TOP: label = "Base top"; break;
    }
    std::vector<cv::Point> points;
    for (const auto& point : result.points_2d_)
        points.emplace_back(cvRound(point.x), cvRound(point.y));
    draw_box(img, points, label);
}

void draw_detection(cv::Mat& img, const backend::types::RuneResult& result)
{
    const char* label = "Unknown";
    using backend::types::RuneType;
    switch (result.class_id)
    {
        case RuneType::UNACTIVATED: label = "Unactivated"; break;
        case RuneType::SMALL_ACTIVATED: label = "Small activated"; break;
        case RuneType::LARGE_ACTIVATED: label = "Large activated"; break;
    }
    std::vector<cv::Point> points;
    for (const auto& point : {result.top, result.left, result.right, result.bottom})
        points.emplace_back(cvRound(point.x), cvRound(point.y));
    draw_box(img, points, label);
    cv::circle(img, result.point_R, 4, cv::Scalar(0, 0, 255), cv::FILLED, cv::LINE_AA);
}

struct IMG
{
    cv::Mat img;
    int frame_count;
};

class ImgQueue
{
public:
    ImgQueue(const std::string& input_path, size_t queue_size, bool print_video_fps = false)
    : queue_size_(queue_size), print_video_fps_(print_video_fps)
    {
        if (queue_size_ == 0)
            backend::tools::logger() -> error("数据队列长度不能为0！");
        video_ = cv::VideoCapture(input_path);
        thread_ = std::thread([this]() { loop(); });
    }

    ~ImgQueue()
    {
        close();
    }

    void close()
    {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            close_ = true;
        }
        conditional_.notify_all();
        if (thread_.joinable())
        {
            thread_.join();
            if (print_video_fps_ && decode_delay_seconds_ > 0.0)
                backend::tools::logger()->info(
                    "视频输出FPS（解码+delay，不含队列等待）: {:.2f}",
                    decoded_frames_ / decode_delay_seconds_);
        }
    }

    std::optional<IMG> get_IMG()
    {
        std::unique_lock<std::mutex> lock(mutex_);
        conditional_.wait(lock, [this](){
           return close_ || !queue_.empty();
        });

        if (queue_.empty()) return std::nullopt;

        auto img = std::move(queue_.front());
        queue_.pop();
        lock.unlock();
        conditional_.notify_one();
        return img;
    }

private:
    cv::VideoCapture video_;
    std::queue<IMG> queue_;
    size_t queue_size_;
    bool print_video_fps_;
    size_t decoded_frames_ = 0;
    double decode_delay_seconds_ = 0.0;
    bool close_ = false;

    std::thread thread_;
    std::mutex mutex_;
    std::condition_variable conditional_;

private:
    void loop()
    {
        for (int frame = 0; ; ++frame)
        {
            {
                std::unique_lock<std::mutex> lock(mutex_);
                conditional_.wait(lock, [this]() {
                    return close_ || queue_.size() < queue_size_;
                });
                if (close_) return;
            }

            IMG img_;
            auto start = print_video_fps_ ? std::chrono::steady_clock::now()
                                         : std::chrono::steady_clock::time_point{};
            if (!video_.read(img_.img) || img_.img.empty())
            {
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    close_ = true;
                }
                conditional_.notify_all();
                return;
            }
            const double decode_seconds = print_video_fps_
                ? std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()
                : 0.0;
            img_.frame_count = frame;

            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (close_) return;
                queue_.push(std::move(img_));
            }
            conditional_.notify_one();
            if (print_video_fps_) start = std::chrono::steady_clock::now();
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            if (print_video_fps_)
            {
                decode_delay_seconds_ += decode_seconds + std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - start).count();
                ++decoded_frames_;
            }
        }
    }
};

int main(int argc, char* argv[])
{
    // 终端输入处理
    cv::CommandLineParser cli(argc, argv, keys);
    if (cli.has("help"))
    {
        cli.printMessage();
        return 0;
    }

    auto input_path = cli.get<std::string>(0);
    auto config_path = cli.get<std::string>("config-path");
    auto start_index = cli.get<int>("start_index");
    auto end_index = cli.get<int>("end_index");
    auto queue_size = cli.get<size_t>("queue_size");
    auto display = cli.get<bool>("display");
    auto print_video_fps = cli.get<bool>("print_video_fps");

    if (!cli.check())
    {
        cli.printErrors();
        return 1;
    }
    if (start_index < 0 || end_index < 0 ||
        (end_index != 0 && end_index < start_index) || queue_size == 0)
    {
        backend::tools::logger()->error("帧范围无效或队列长度为0");
        return 1;
    }

    auto config = YAML::LoadFile(config_path);
    auto pipeline = config["pipeline"];
    auto mode = pipeline["inference"]["mode"].as<std::string>("sync");
    if (mode != "async" && mode != "sync")
    {
        backend::tools::logger()->error("未知推理模式: {}", mode);
        return 1;
    }
    backend::Detector detector(pipeline);
    ImgQueue img_queue(input_path, queue_size, print_video_fps);

    backend::types::DetectionResults result;
    double inference_seconds = 0.0;
    size_t processed_frames = 0;
    const auto start = std::chrono::steady_clock::now();

    while (auto frame = img_queue.get_IMG())
    {
        if (frame->frame_count < start_index) continue;
        if (end_index != 0 && frame->frame_count > end_index) break;

        auto& img = frame->img;
        const auto frame_count = frame->frame_count;
        auto f = [&]() {
            if (mode == "async")
            {
                auto pending = detector.submit(img, frame_count);
                return detector.collect(pending);
            }
            return detector.detect(img, frame_count);
        };

        inference_seconds += calculate_cost(f, result);
        ++processed_frames;

        bool stop = false;
        if (display)
        {
            std::visit([&img](const auto& detections) {
                for (const auto& detection : detections)
                    draw_detection(img, detection);
            }, result);
            cv::imshow("test_fps", img);
            const auto key = cv::waitKey(1);
            stop = key == 27 || key == 'q';
        }
        if (stop || (end_index != 0 && frame_count == end_index)) break;
    }

    const auto elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    img_queue.close();
    if (display) cv::destroyAllWindows();

    const double inference_fps = inference_seconds > 0.0
        ? processed_frames / inference_seconds : 0.0;
    const double overall_fps = elapsed > 0.0 ? processed_frames / elapsed : 0.0;
    backend::tools::logger()->info(
        "测试结束: 处理 {} 帧，推理累计耗时 {:.3f} s，推理 FPS {:.2f}，"
        "整体耗时 {:.3f} s，整体 FPS {:.2f}",
        processed_frames, inference_seconds, inference_fps, elapsed, overall_fps);
    return 0;
}
