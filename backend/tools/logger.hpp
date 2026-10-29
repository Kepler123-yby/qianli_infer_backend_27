#pragma once

#include <spdlog/spdlog.h>

#include <memory>

namespace backend::tools
{

// 线程安全的共享日志：控制台输出info及以上，文件保存debug及以上。
std::shared_ptr<spdlog::logger> logger();

} // namespace backend::tools
