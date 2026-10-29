#include "logger.hpp"

#include <fmt/chrono.h>
#include <spdlog/sinks/ostream_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>

namespace backend::tools
{

std::shared_ptr<spdlog::logger> logger()
{
    // 文件先于logger构造、后于logger销毁，保证ostream_sink引用始终有效。
    static std::ofstream log_file;

    // 局部static保证多个推理线程同时首次访问时只初始化一次。
    static const auto instance = []
    {
        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        console_sink->set_level(spdlog::level::info);

        auto result = std::make_shared<spdlog::logger>("backend", console_sink);
        result->set_level(spdlog::level::debug);
        result->flush_on(spdlog::level::info);

        const auto file_name = fmt::format("logs/{:%Y-%m-%d_%H-%M-%S}.log",
                                          std::chrono::system_clock::now());
        std::error_code error;
        std::filesystem::create_directories("logs", error);
        if (!error) log_file.open(file_name, std::ios::out | std::ios::app);

        if (log_file.is_open())
        {
            auto file_sink = std::make_shared<spdlog::sinks::ostream_sink_mt>(log_file);
            file_sink->set_level(spdlog::level::debug);
            result->sinks().push_back(file_sink);
        }

        else
        {
            result->error("日志文件无法创建，仅输出控制台: {}，{}", file_name,
                          error ? error.message() : "文件打开失败");
        }

        return result;
    }();

    return instance;
}

} // namespace backend::tools
