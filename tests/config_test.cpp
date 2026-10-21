#include "backend.hpp"
#include "processing/processor.hpp"
#include "settings/pipeline_setting.hpp"
#include "settings/yaml_utils.hpp"

#include <cassert>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <type_traits>

namespace
{

void expectRejected(const std::string& yaml)
{
    try
    {
        const backend::setting::PipelineConfig config(YAML::Load(yaml));
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
    "inference: {model_path: model.engine, type: tensorrt}\n";

} // namespace

int main()
{
    using namespace backend;
    static_assert(std::is_constructible_v<Detector, const setting::PipelineConfig&>);
    static_assert(!std::is_constructible_v<Detector, const YAML::Node&>);
    static_assert(std::is_constructible_v<processing::Processor, const setting::ProcessingConfig&>);
    static_assert(!std::is_constructible_v<processing::Processor, const YAML::Node&>);

    const setting::PipelineConfig defaults(YAML::Load(required));
    assert(defaults.inference.model_path == "model.engine");
    assert(defaults.processing.postprocess.decoder == "yolo_v8");
    assert(defaults.inference.type == setting::InferBackend::TENSORRT);
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
    assert(defaults.inference.mode == setting::InferMode::SYNC);
    assert(defaults.inference.input_capacity == 1);
    assert(defaults.inference.result_capacity == 1);
    assert(defaults.inference.overflow == "drop_oldest");

    assert(tools::readConfig<int>(YAML::Load("{}"), "value", 7) == 7);
    assert(tools::readConfig<int>(YAML::Load("value: null"), "value", 7) == 7);
    assert(tools::readConfig<int>(YAML::Load("value: 9"), "value", 7) == 9);

    const setting::PipelineConfig null_sections(YAML::Load(required + "processing: null\n"));
    assert(!null_sections.processing.preprocess.use_cuda);
    assert(null_sections.inference.mode == setting::InferMode::SYNC);
    const setting::PipelineConfig null_values(YAML::Load(R"(
inference: {model_path: model.engine, type: tensorrt, mode: null, input_capacity: null}
processing:
  preprocess: {implementation: null, center_padding: null}
  postprocess: {decoder: null, confidence_threshold: null}
)"));
    assert(!null_values.processing.preprocess.use_cuda);
    assert(null_values.processing.preprocess.use_center_padding);
    assert(null_values.processing.postprocess.decoder == "yolo_v8");
    assert(null_values.processing.postprocess.confidence_threshold == 0.5);
    assert(null_values.inference.mode == setting::InferMode::SYNC);
    assert(null_values.inference.input_capacity == 1);

    const auto example = setting::loadPipelineConfig("config/default.yaml");
    assert(example.processing.preprocess.use_cuda);
    assert(example.processing.postprocess.use_cuda);
    assert(example.inference.type == setting::InferBackend::TENSORRT);
    assert(example.inference.mode == setting::InferMode::ASYNC);
    assert(std::filesystem::is_regular_file(example.inference.model_path));
    const Detector pipeline(example);
    const processing::Processor processor(example.processing);

    const setting::PipelineConfig custom(YAML::Load(R"(
processing:
  preprocess:
    implementation: cpu
    center_padding: false
    padding_color: [0, 114, 255]
    normalize_scale: 0.5
  postprocess:
    decoder: yolo_v5
    implementation: cuda
    confidence_threshold: 0.25
    nms_threshold: 0.6
inference:
  model_path: custom.xml
  type: openvino
  openvino: {device: GPU}
  tensorrt: ignored_unselected_branch
  mode: async
  input_capacity: 2
  result_capacity: 3
  overflow: drop_oldest
)"));
    assert(custom.processing.postprocess.decoder == "yolo_v5");
    assert(custom.inference.model_path == "custom.xml");
    assert(!custom.processing.preprocess.use_center_padding);
    assert(custom.processing.preprocess.padding_color == cv::Scalar(0, 114, 255));
    assert(custom.processing.preprocess.normalize_scale == 0.5);
    assert(custom.processing.postprocess.confidence_threshold == 0.25);
    assert(custom.processing.postprocess.nms_threshold == 0.6);
    assert(custom.inference.type == setting::InferBackend::OPENVINO);
    assert(custom.inference.openvino.device == "GPU");
    assert(custom.inference.input_capacity == 2);
    assert(custom.inference.result_capacity == 3);

    const setting::PipelineConfig trt(YAML::Load(
        "inference: {model_path: custom.engine, type: tensorrt, tensorrt: {device_id: 2}, openvino: ignored}\n"));
    assert(trt.inference.tensorrt.device_id == 2);
    const setting::PipelineConfig ov(YAML::Load(
        "inference: {model_path: custom.xml, type: openvino}\n"));
    assert(ov.inference.openvino.device == "CPU");

    expectRejected("{}");
    expectRejected("inference: {type: tensorrt}");
    expectRejected("inference: {model_path: '', type: tensorrt}");
    expectRejected("inference: {model_path: null, type: tensorrt}");
    expectRejected("inference: {model_path: x}");
    expectRejected("inference: {model_path: x, type: unknown}");
    expectRejected("inference: wrong");
    for (const auto& section : {"processing: wrong",
                               "processing: {preprocess: 1}", "processing: {postprocess: 1}"})
        expectRejected(required + section);
    for (const auto& field : {"center_padding: wrong", "padding_color: wrong",
                             "normalize_scale: wrong"})
        expectRejected(required + "processing:\n  preprocess: {" + field + "}\n");
    for (const auto& field : {"confidence_threshold: wrong", "nms_threshold: wrong"})
        expectRejected(required + "processing:\n  postprocess: {" + field + "}\n");
    for (const auto& field : {"mode: unknown", "input_capacity: wrong", "result_capacity: wrong",
                             "tensorrt: {device_id: wrong}"})
        expectRejected("inference: {model_path: model.engine, type: tensorrt, " +
                       std::string(field) + "}\n");

    std::cout << "PASS: hierarchical config, defaults, overrides, backend selection and invalid inputs\n";
}
