#include "inference/inference.hpp"

#include <openvino/opsets/opset13.hpp>
#include <openvino/core/graph_util.hpp>

#include <algorithm>
#include <cassert>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>

using backend::inference::Inference;
using backend::setting::InferBackend;
using backend::setting::InferMode;
using backend::setting::InferenceConfig;

// 通过config.type分发到OpenVINO后端的CPU集成测试；argv[1]为临时模型目录。
int main(int argc, char** argv)
{
    if (argc != 2)
        throw std::runtime_error("请提供临时测试目录");
    const std::filesystem::path directory(argv[1]);
    std::filesystem::create_directories(directory);

    auto parameter = std::make_shared<ov::opset13::Parameter>(
        ov::element::f32, ov::Shape{1, 1, 2, 2});
    auto relu = std::make_shared<ov::opset13::Relu>(parameter);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{relu}, ov::ParameterVector{parameter});
    const auto model_path = (directory / "relu.xml").string();
    ov::serialize(model, model_path);

    InferenceConfig config;
    config.model_path = model_path;
    config.type = InferBackend::OPENVINO;
    config.mode = InferMode::ASYNC;
    config.openvino.device = "CPU";
    config.worker = 2;
    config.capacity = 1;

    const int shape[] = {1, 1, 2, 2};
    cv::Mat input(4, shape, CV_32F);
    const float values[] = {-1, 2, -3, 4};
    std::copy(std::begin(values), std::end(values), input.ptr<float>());
    const std::vector<float> expected{0, 2, 0, 4};

    {
        Inference inference(config);
        assert(inference.type() == InferBackend::OPENVINO);
        assert(inference.mode() == InferMode::ASYNC);

        // 同步接口与异步接口走同一后端，配置只决定上层用哪一个。
        assert(inference.infer(input) == expected);
        assert(inference.inferAsync(input).get() == expected);

        // 移动后仍可继续推理，说明后端所有权随移动转移。
        Inference moved(std::move(inference));
        assert(moved.infer(input) == expected);
    }

    // 非法输入由后端检查，错误经Inference转发给调用方。
    {
        Inference inference(config);
        bool threw = false;
        try
        {
            inference.infer(cv::Mat(2, 2, CV_8U));
        }
        catch (const std::runtime_error&)
        {
            threw = true;
        }
        assert(threw);
    }

    // OpenVINO不支持动态形状，构造阶段必须失败。
    parameter->set_partial_shape(ov::PartialShape{-1, 1, 2, 2});
    model->validate_nodes_and_infer_types();
    config.model_path = (directory / "dynamic.xml").string();
    ov::serialize(model, config.model_path);
    {
        bool threw = false;
        try
        {
            Inference inference(config);
        }
        catch (const std::exception&)
        {
            threw = true;
        }
        assert(threw);
    }

    std::cout << "PASS: config.type分发、同步/异步转发、移动语义、错误转发\n";
}
