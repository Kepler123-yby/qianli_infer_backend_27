#include "types/detection_result.hpp"

namespace backend::postprocess
{

class YOLOBase
{
public:

    /**
     * @brief yolo后处理操作
     * 
     * @param output 推理得到的最终结果
     * @param frame_count 当前图片的计数
     * @return  YOLO识别结果
     */
    virtual types::YOLOResult postprocess(const std::vector<float>& output, int frame_count);

};
    
} // namespace backend::postprocess