# qianli_infer_backend_27
重庆大学千里战队27赛季推理后端

## 目录结构

```text
backend/
├── backend.hpp / .cpp            # 对外接口
├── settings/                    # 配置类型与 YAML 读取工具
├── types/                       # 检测结果类型
├── inference/                   # 推理后端
│   ├── openvino/
│   └── tensorrt/
└── processing/                  # 前后处理
    ├── processor.hpp / .cpp
    ├── cpu/
    └── cuda/
config/                          # YAML 配置文件
models/                          # 模型资源（原 asserts/）
```

模型路径相对于工作目录；以下示例和测试以项目根目录为工作目录。

## 分层配置

`config/default.yaml` 使用 `pipeline` 根节点，由 `PipelineConfig` 按键分发：

- `model` → `ModelConfig`：模型名称与路径。
- `processing` → `ProcessingConfig`：继续分发 `preprocess`、`postprocess`。
- `inference` → `InferenceConfig`：推理后端、同步/异步模式及队列参数；仅解析对应的 `tensorrt` 或 `openvino` 子节点。

`settings/` 下文件统一使用 `*_setting` 命名；前后处理配置集中在
`processing_setting.hpp/.cpp`，不再单独拆分预处理配置文件。

上层直接通过 `config["键"]` 分发节点；`yaml_utils.hpp` 只保留 `readConfig()`，
键缺失或值为 `null` 时使用默认值，类型转换由 yaml-cpp 处理。
模型路径等必填项和参数取值仍由各配置结构体检查。

