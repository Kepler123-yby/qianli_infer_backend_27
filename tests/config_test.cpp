#include "backend.hpp"
#include "processing/processor.hpp"
#include "settings/pipeline_config.hpp"

#include <cassert>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <type_traits>

namespace
{

void expectRejected(const std::string& yaml)
{
    try
    {
        const backend::PipelineConfig config(YAML::Load(yaml));
    }
    catch (const std::invalid_argument&)
    {
        return;
    }
    catch (const YAML::Exception&)
    {
        return;
    }
    assert(false && "Invalid configuration must be rejected");
}

const std::string required =
    "model: {path: model.engine}\n"
    "inference: {type: tensorrt}\n";

} // namespace

int main()
{
    using namespace backend;
    static_assert(std::is_constructible_v<YOLOModel, const ModelConfig&>);
    static_assert(!std::is_constructible_v<YOLOModel, const YAML::Node&>);
    static_assert(std::is_constructible_v<Processor, const ProcessingConfig&>);
    static_assert(!std::is_constructible_v<Processor, const YAML::Node&>);

    const PipelineConfig defaults(YAML::Load(required));
    assert(defaults.model.model_name == "yolov8");
    assert(defaults.model.model_path == "model.engine");
    assert(defaults.inference.type == InferBackend::TENSORRT);
    assert(defaults.inference.tensorrt.device_id == 0);
    assert(!defaults.processing.preprocess.use_cuda);
    assert(!defaults.processing.postprocess.use_cuda);
    assert(defaults.processing.preprocess.use_center_padding);
    assert(defaults.processing.preprocess.padding_color == cv::Scalar(124, 124, 124));
    assert(defaults.processing.preprocess.normalize_scale == 1.0 / 255.0);
    assert(defaults.processing.preprocess.width == 0);
    assert(defaults.processing.preprocess.height == 0);
    assert(defaults.processing.postprocess.confidence_threshold == 0.5);
    assert(defaults.processing.postprocess.nms_threshold == 0.45);
    assert(defaults.execution.mode == InferMode::SYNC);
    assert(defaults.execution.input_capacity == 1);
    assert(defaults.execution.result_capacity == 1);
    assert(defaults.execution.overflow == "drop_oldest");

    const auto example = loadPipelineConfig("config/default.yaml");
    assert(example.processing.preprocess.use_cuda);
    assert(example.processing.postprocess.use_cuda);
    assert(example.inference.type == InferBackend::TENSORRT);
    assert(example.execution.mode == InferMode::ASYNC);
    assert(std::filesystem::is_regular_file(example.model.model_path));
    const Processor processor(example.processing);

    const PipelineConfig custom(YAML::Load(R"(
model: {name: custom, path: custom.xml}
processing:
  preprocess:
    implementation: cpu
    center_padding: false
    padding_color: [0, 114, 255]
    normalize_scale: 0.5
  postprocess:
    implementation: cuda
    confidence_threshold: 0.25
    nms_threshold: 0.6
inference:
  type: openvino
  openvino: {device: GPU}
  tensorrt: ignored_unselected_branch
execution:
  mode: async
  input_capacity: 2
  result_capacity: 3
  overflow: drop_oldest
)"));
    assert(custom.model.model_name == "custom");
    assert(custom.model.model_path == "custom.xml");
    assert(!custom.processing.preprocess.use_center_padding);
    assert(custom.processing.preprocess.padding_color == cv::Scalar(0, 114, 255));
    assert(custom.processing.preprocess.normalize_scale == 0.5);
    assert(custom.processing.postprocess.confidence_threshold == 0.25);
    assert(custom.processing.postprocess.nms_threshold == 0.6);
    assert(custom.inference.type == InferBackend::OPENVINO);
    assert(custom.inference.openvino.device == "GPU");
    assert(custom.execution.input_capacity == 2);
    assert(custom.execution.result_capacity == 3);

    const PipelineConfig trt(YAML::Load(
        "model: {path: custom.engine}\n"
        "inference: {type: tensorrt, tensorrt: {device_id: 2}, openvino: ignored}\n"));
    assert(trt.inference.tensorrt.device_id == 2);
    const PipelineConfig ov(YAML::Load(
        "model: {path: custom.xml}\ninference: {type: openvino}\n"));
    assert(ov.inference.openvino.device == "CPU");

    expectRejected("{}");
    expectRejected("model: {}\ninference: {type: tensorrt}");
    expectRejected("model: {path: ''}\ninference: {type: tensorrt}");
    expectRejected("model: {path: null}\ninference: {type: tensorrt}");
    expectRejected("model: {path: x}\ninference: {}");
    expectRejected("model: {path: x}\ninference: {type: unknown}");
    expectRejected("model: {path: x}\ninference: {type: tensorrt, tensorrt: {device_id: -1}}");
    expectRejected("model: {path: x}\ninference: {type: openvino, openvino: {device: ''}}");
    expectRejected("model: {path: x}\ninference: {type: openvino, openvino: []}");
    for (const auto& section : {"processing: null", "processing: []", "execution: []",
                               "processing: {preprocess: 1}", "processing: {postprocess: null}"})
        expectRejected(required + section);
    for (const auto& field : {"implementation: unknown", "implementation: null",
                             "center_padding: wrong", "center_padding: null",
                             "padding_color: [1, 2]", "padding_color: [0, 0, 256]",
                             "padding_color: [0, .nan, 0]", "padding_color: wrong",
                             "normalize_scale: -1", "normalize_scale: .inf"})
        expectRejected(required + "processing:\n  preprocess: {" + field + "}\n");
    for (const auto& field : {"implementation: unknown", "confidence_threshold: 1.1",
                             "confidence_threshold: .nan", "nms_threshold: -1",
                             "nms_threshold: wrong"})
        expectRejected(required + "processing:\n  postprocess: {" + field + "}\n");
    for (const auto& field : {"mode: unknown", "input_capacity: 0", "result_capacity: -1",
                             "input_capacity: wrong", "overflow: drop_newest"})
        expectRejected(required + "execution: {" + field + "}\n");

    std::cout << "PASS: hierarchical config, defaults, overrides, backend selection and invalid inputs\n";
}
