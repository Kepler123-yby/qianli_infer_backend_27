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

- `processing` → `ProcessingConfig`：继续分发 `preprocess`、`postprocess`；YOLO 输出解析名称放在 `postprocess.decoder`。
- `inference` → `InferenceConfig`：模型路径、推理后端、同步/异步模式及队列参数；仅解析对应的 `tensorrt` 或 `openvino` 子节点。

顶层 `Inference` 接收 `PipelineConfig`，分别传递处理配置和保存推理配置，
不再保留 `YOLOModel` 中间层或独立的模型配置。

`settings/` 下文件统一使用 `*_setting` 命名；前后处理配置集中在
`processing_setting.hpp/.cpp`，不再单独拆分预处理配置文件。

上层直接通过 `config["键"]` 分发节点；`yaml_utils.hpp` 只保留 `readConfig()`，
键缺失或值为 `null` 时使用默认值，类型转换由 yaml-cpp 处理。
模型路径不能为空；后端类型与执行模式沿用现有解析检查。
当前仅接通配置传递，`decoder` 只记录解析器名称，尚未实现模型执行、YOLO 解码或异步队列。

