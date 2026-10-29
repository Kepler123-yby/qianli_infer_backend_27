#pragma once

#include "settings/pipeline_setting.hpp"
#include "types/detection_result.cpp"

#include <cstdint>
#include <vector>

namespace backend::processing
{

/**
 * @brief 后处理对外接口，按配置选择解码器
 */
class PostBase
{
public:
    explicit PostBase(const setting::PostprocessConfig& config);
    PostBase() = default;

    /**
     * @brief 解码一帧推理输出
     *
     * @param output 模型原始输出
     * @param rp_scale 本帧预处理使用的缩放和填充参数，不能跨帧缓存
     * @param frame_count 输入端分配的帧号
     * @return 检测结果；无目标、参数无效或解码失败时返回空数组
     */
    std::vector<types::YOLOResult> process(const std::vector<float>& output,
                                           const setting::ResizePadding& rp_scale,
                                           std::uint64_t frame_count);

private:
    setting::PostprocessConfig config_;
};

} // namespace backend::processing
