#include "postprocess.hpp"
#include "tools/logger.hpp"

#include "decode/armor_v5.cpp"
#include "decode/armor_v8.cpp"

namespace backend::processing
{

PostBase::PostBase(const setting::PostprocessConfig& config) : config_(config)
{
}

std::vector<types::YOLOResult> PostBase::process(const std::vector<float>& output,
                                                 const setting::ResizePadding& rp_scale,
                                                 std::uint64_t frame_count)
{
    // 解码器只保存配置和本帧缩放参数，构造代价很小，不需要跨帧缓存。
    if (config_.decoder == "yolo_v5")
    {
        postprocess::ArmorV5 decoder(config_, rp_scale);
        return decoder.postprocess(output, frame_count);
    }

    if (config_.decoder == "yolo_v8")
    {
        postprocess::ArmorV8 decoder(config_, rp_scale);
        return decoder.postprocess(output, frame_count);
    }

    tools::logger()->error("[postprocess] 不支持的解码器: {}，跳过本帧", config_.decoder);
    return {};
}

} // namespace backend::processing
