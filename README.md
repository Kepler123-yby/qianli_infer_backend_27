# qianli_infer_backend_27

重庆大学千里战队 27 赛季推理后端。

## 目录

- [1. 项目简介](#1-项目简介)
- [2. 环境与依赖](#2-环境与依赖)
- [3. 快速开始](#3-快速开始)
- [4. 配置说明](#4-配置说明)
- [5. 模型准备与加载](#5-模型准备与加载)
- [6. 接口接入](#6-接口接入)
- [7. 测试与可视化](#7-测试与可视化)
- [8. 目录结构与常见问题](#8-目录结构与常见问题)
- [9. 参考项目与致谢](#9-参考项目与致谢)

## 1. 项目简介

核心部分为`backend`,该项目是面向 RoboMaster 视觉任务的 C++ 神经网络推理库。项目封装cpu/cuda预处理、tensorrt/openvino推理后端(包含同步、异步两种实现)、cpu后处理，当前支持v8/v5装甲板四点模型以及能量机关模型的解码，是一个端到端的推理后端。

## 2. 环境与依赖

本项目支持依赖标准为：Ubuntu 22.04、C++20、CMake3.28、OpenCV4.13(CUDA可选)、TensorRT 10.10.0.31、OpenVINO 2024.6.0、CUDA Toolkit-11.8

**部署方式如下：**

### OpenVINO
[官方部署教程](https://github.com/openvinotoolkit/openvino/blob/2026.4.1/docs/dev/build.md)

### CUDA Toolkit 部署
[官方部署教程](https://developer.nvidia.com/cuda-downloads?target_os=Linux)

### OpenCV 部署 
[官方部署教程](https://docs.opencv.ac.cn/4.13.0/d7/d9f/tutorial_linux_install.html)

### OpenCV CUDA 部署（可选）

完整步骤见 [OpenCV CUDA 部署文档](docs/opencv-cuda.md)，包含驱动与 Toolkit 检查、
可选 cuDNN、编译依赖、源码构建、环境配置及本地安装验证。

### TensorRT 部署

完整步骤见 [TensorRT 部署文档](docs/tensorrt.md)，包含 Debian 本地仓库与 Tar 安装、
环境配置、工具与动态库依赖验证。

**如使用`jetson nrin nx`，则只需根据`OpenCV CUDA`教程来配置**

## 3. 快速开始

以下运行示例应以项目根目录为工作目录，模型路径相对于工作目录解析。

```bash
cmake -S . -B build
cmake --build build -j$(nproc)

build/test_fps
```

## 4. 配置说明

`configs/default.yaml` 使用 `pipeline` 根节点，由 `PipelineConfig` 按键分发：

- `processing` → `ProcessingConfig`：包含 `preprocess`、`postprocess`；解码器名称由 `postprocess.decoder` 指定。
- `inference` → `InferenceConfig`：包含模型路径、推理后端、同步/异步模式及队列参数，以及对应的 `tensorrt` 或 `openvino` 子节点。

如需更改相关设置，更改configs/目录下文件

## 5. 模型准备与加载

### 模型文件与解码器

仓库的 `assets/` 已包含以下模型。使用自定义模型时，需保持与对应解码器一致的输出布局、类别定义和关键点顺序；
通用目标检测 YOLO 模型不能仅凭版本号直接替换装甲板或符文模型。

| 模型 | 文件名前缀 | `pipeline.processing.postprocess.decoder` |
|---|---|---|
| YOLO v5 装甲板四点模型 | `armor_v5n_fp16` | `yolo_v5` |
| YOLO v8 装甲板四点模型 | `armor_v8n_fp16` | `yolo_v8` |
| YOLO v8 符文关键点模型 | `rune_v8n_fp16` | `rune_v8` |

每个前缀对应的模型文件用途如下：

| 文件格式 | 用途 |
|---|---|
| `.onnx` | 供 TensorRT 解析并构建 engine；OpenVINO 也可通过 `read_model()` 加载兼容的 ONNX |
| `.engine` | TensorRT 序列化执行计划，加载时反序列化，不在此阶段切换计算精度 |
| `.xml` + `.bin` | OpenVINO IR；XML 描述网络，BIN 保存权重，使用时保留同目录、同名前缀的配套文件，配置指向 XML |

`model_path` 相对于程序运行时的工作目录解析，不是相对于 YAML 文件所在目录。
下面的命令均在项目根目录执行，也可以将模型路径改成绝对路径。

当前两个后端均要求单输入、单输出、静态形状，输入为 `[1, 3, H, W]`，输入输出张量均为 FP32。
TensorRT 还要求输入输出为线性布局。检测接口接收 BGR 图像，由预处理完成缩放填充、BGR 转 RGB、
归一化和 NCHW 转换；归一化、填充参数需要与模型训练及导出方式匹配。

### TensorRT 加载与缓存

在现有完整配置中修改以下字段，其他前后处理和队列设置保留：

```yaml
pipeline:
  processing:
    postprocess:
      decoder: yolo_v8
  inference:
    type: tensorrt
    model_path: assets/armor_v8n_fp16.onnx
    tensorrt:
      device_id: 0
```

设置为 `.onnx` 时：

1. 查找同目录、同名前缀的 `.engine`，例如 `armor_v8n_fp16.onnx` 对应 `armor_v8n_fp16.engine`。
2. 缓存文件存在且修改时间不早于 ONNX 时，直接读取并反序列化。
3. 缓存缺失或早于 ONNX 时，在当前 GPU 上构建 engine，写入同名缓存并加载。构建会阻塞初始化，模型目录需有写入权限。

代码构建 engine 时已显式启用 `BuilderFlag::kFP16`，允许内部采用 FP16，也允许 FP32 回退；
输入输出类型沿用模型定义。缓存有效性仅比较文件时间，不记录 GPU、TensorRT 版本或构建选项。
更换部署环境或构建选项后，如需强制重建，应先移走旧 engine，再以 ONNX 路径启动。

也可以将 `model_path` 直接设为 `assets/armor_v8n_fp16.engine`：

- 读取并反序列化成功时直接使用，不检查它是否早于 ONNX。
- 文件读取失败、数据为空或反序列化失败时，查找同目录的 `armor_v8n_fp16.onnx`。
- 找到 ONNX 后重新构建并尝试加载，成功后覆盖原 engine 文件；ONNX 缺失或重建失败时结束初始化。

该回退只针对直接加载 engine 的路径。配置为 ONNX 时，如果命中的缓存无法读取或反序列化，
当前实现会结束初始化，需要手动移走损坏缓存后重试。engine 已反序列化但其形状、布局或类型不符合上述接口要求时，也不会触发回退。

### 使用命令行构建 FP16 engine

完成 [TensorRT 本地部署](docs/tensorrt.md) 后，可使用 TensorRT 10.x 的 `trtexec` 显式构建：

```bash
trtexec \
  --onnx=assets/armor_v8n_fp16.onnx \
  --saveEngine=assets/armor_v8n_fp16_local.engine \
  --fp16 \
  --inputIOFormats=fp32:chw \
  --outputIOFormats=fp32:chw \
  --profilingVerbosity=detailed \
  --skipInference
```

`--fp16` 允许 FP16 内部计算，`fp32:chw` 保持输入输出为 FP32 线性布局，匹配当前执行路径。
`--skipInference` 只构建并保存，`--profilingVerbosity=detailed` 将详细层信息保存在 engine 中。
输出使用独立文件名，使用时将 `model_path` 改为 `assets/armor_v8n_fp16_local.engine`。
此文件的自动回退会查找同名的 `armor_v8n_fp16_local.onnx`；若需要保留原模型的同名回退关系，应自行统一文件前缀。

优先在实际部署的 GPU 和 TensorRT 环境中构建 engine。仓库中已有的 engine 不保证能直接用于另一台设备；
模型迁移时保留 ONNX，按部署环境重新构建。

### OpenVINO 加载

切换到 OpenVINO 时，在完整配置中修改以下字段；装甲板 v8 的解码器仍为 `yolo_v8`：

```yaml
pipeline:
  processing:
    preprocess:
      implementation: cpu
    postprocess:
      decoder: yolo_v8
  inference:
    type: openvino
    model_path: assets/armor_v8n_fp16.xml
    openvino:
      device: CPU
```

确认 `assets/armor_v8n_fp16.bin` 与 XML 配套存在。后端通过 `read_model()` 读取模型，
随后按 `openvino.device` 编译并创建推理请求；这里选择 CPU 预处理和 CPU 推理，预处理与推理设备分别配置。
若改用 ONNX，将路径指向相应 `.onnx`，其输入输出仍须满足前述要求。

### 区分与验证精度

模型权重类型、内部计算精度、输入输出类型是三个不同概念。文件名包含 `fp16` 不能证明所有层都使用 FP16，
FP32 输入输出也不代表内部计算全为 FP32。OpenVINO 的 FP16 权重 IR 同样可以具有 FP32 输入输出。

检查已生成 engine 的层信息：

```bash
trtexec \
  --loadEngine=assets/armor_v8n_fp16_local.engine \
  --dumpLayerInfo \
  --exportLayerInfo=/tmp/armor_v8_layers.json \
  --skipInference
```

结合各层的类型、格式和 tactic 信息判断精度。若只返回层名，说明构建时没有保存详细信息，
需在重新构建时启用 `--profilingVerbosity=detailed`；加载时追加该选项不能补回缺失的信息。
代码内部自动构建的 engine 当前未启用详细层信息。更多说明见
[NVIDIA Engine Inspector 文档](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/engine-tools.html)。

## 6. 接口接入

如要使用该后端，复制backend/到自己的项目目录下，之后在`CMakeLists.txt`中按如下添加：

```cmake
add_subdirectory(backend)
target_link_libraries(target_project PRIVATE backend ...)
```

**最小使用案例：**

**同步：**

```cpp
Detector detector;
auto result = detector.detect(img, frame_count);
```

**异步：**

```cpp
Detector detector;
auto pending = detector.submit(img, frame_count);
auto result = detector.collect();
```

## 7. 测试与可视化

`build/test_fps` 是当前唯一的测试程序：从视频逐帧解码、调用 `Detector` 检测、可选地画出结果，结束时打印 FPS。项目没有接入 CTest，验证方式就是直接运行该程序并核对日志。

### 命令行参数

```bash
# 使用默认视频与 configs/test_demo.yaml
build/test_fps

# 测前200帧并打开可视化窗口
build/test_fps assets/video/装甲板.mp4 --end_index=199 --display=1

# 换队列长度，同时打印解码线程的输出FPS
build/test_fps assets/video/装甲板.mp4 --start_index=100 --end_index=199 --qs=16 --print_video_fps=true
```

| 参数 | 默认值 | 用途 |
|---|---|---|
| `@input-path`（位置参数） | `assets/video/装甲板.mp4` | 测试视频路径 |
| `config-path` / `c` | `configs/test_demo.yaml` | 配置文件路径 |
| `start_index` / `s` | `0` | 起始帧号（含） |
| `end_index` / `e` | `0` | 结束帧号（含）；`0` 表示读到视频末尾 |
| `queue_size` / `qs` | `8` | 解码线程与检测端之间的帧队列长度 |
| `display` / `d` | `0` | `1` 打开可视化窗口 |
| `print_video_fps` | `false` | `true` 时额外打印解码线程的输出FPS |
| `help` / `h` |  | 打印参数说明 |

由于 `0` 被用作“读到末尾”的哨兵值，无法只测第 0 帧；测单帧请用非 0 的帧号。

### 运行流程

1. 校验参数：帧号非负，`end_index` 非 0 时不小于 `start_index`，`queue_size` 不为 0，`pipeline.inference.mode` 只能是 `sync` 或 `async`。任一项不满足时不加载模型，直接返回非零。
2. 构造 `Detector` 读取模型，并按模型的输入尺寸初始化预处理。
3. 构造 `ImgQueue` 启动解码线程：从视频起始位置顺序 `read()`，把帧和帧号压入队列；队列满时阻塞等待。每次成功入队后休眠 2 ms，该值写死在代码里，不可配置。
4. 主线程顺序取帧。帧号小于 `start_index` 的帧被丢弃，但已经过解码；取到 `end_index` 那一帧并检测后跳出循环，随后关闭解码线程。解码始终从第 0 帧开始，没有 seek，所以 `start_index` 只影响检测范围，不缩短解码时间。
5. `display=1` 时在取到的原图上画框和标签并 `imshow`，窗口内按 `Esc` 或 `q` 提前结束。

### 输出的两个 FPS

结束时打印一行：处理帧数、推理累计耗时、推理 FPS、整体耗时、整体 FPS。

- **推理 FPS** = 处理帧数 / 推理累计耗时。每帧计时围绕一次 `detect()`（sync）或 `submit()+collect()`（async），包含预处理、推理提交与等待、数据传输和后处理，不是纯 GPU 内核耗时。
- **整体 FPS** = 处理帧数 / 从进入取帧循环到退出循环的墙钟时间。除推理外还含取帧等待、解码线程的供帧节奏、显示开销，以及 `start_index` 之前被丢弃帧的解码时间。

`print_video_fps=true` 时，解码线程关闭时再打印一行「解码+delay」FPS：累计该线程内每次 `read()` 的耗时与随后的 2 ms 休眠，除以成功解码帧数，不含等待队列空位的时间和打开视频的时间。它衡量解码线程的供帧节奏，上限由 2 ms 休眠决定（约 500 FPS），实际更低，因为还要加上 `read()` 本身。

`async` 模式下测试仍是“提交一帧后立即 collect”，同一时刻只有一帧在途，测到的是单帧流水线延迟而不是并发吞吐。需要吞吐数据时，得在测试里改成保持多个 `PendingFrame` 后再批量收集。

### 可视化

`display=1` 时检测结果直接画在取到的原图上，绿色框为四点外接矩形：

- 装甲板模型：按 `TargetType` 标注 `Sentry`、`Hero`、`Engineer`、`Infantry 3/4/5`、`Outpost`、`Base top/bottom`。
- 符文模型：四点外接矩形标注 `Unactivated`、`Small activated`、`Large activated`，并用红点标出 `point_R`。

标签只用于确认解码出的类别与四点位置，不显示置信度。`display=1` 会叠加绘制和 `waitKey(1)` 开销，比较性能时应关闭。

### 读数是否有效

- 推理后端初始化或逐帧推理失败时，后端只打印 error 并在该帧返回空结果，程序依然跑完并照常打印 FPS。发现 `[tensorrt]`、`[openvino]`、`[detector]` 的 error 时，这一轮数字只是解码加空转速度，不能当性能结果。
- 视频路径错误或缺少编解码器时解码线程立即关闭，`processed_frames` 为 0，两个 FPS 都是 0。
- 对比两次运行时固定视频、配置、帧范围、`queue_size` 和 `display`。预处理、推理、后处理的分段耗时当前没有单独统计，需要定位瓶颈时自行在 `detect()` 内部或测试里加计时。

## 8. 目录结构与常见问题

### 目录结构

```text
.
├── CMakeLists.txt                   # 项目构建入口与 test_fps 可执行目标
├── backend/                         # 推理后端库
│   ├── CMakeLists.txt               # 依赖检测与后端库构建
│   ├── detector.hpp / .cpp          # 对外检测接口：detect / submit / collect
│   ├── inference/                   # 推理接口与任务队列
│   │   ├── inference.hpp / .cpp
│   │   ├── openvino/                # OpenVINO 后端
│   │   └── tensorrt/                # TensorRT 后端与 engine 构建、加载
│   ├── processing/                  # 前后处理
│   │   ├── processor.hpp / .cpp     # 前后处理调度
│   │   ├── preprocess/             # 图像预处理，支持 CPU / CUDA
│   │   └── postprocess/            # 后处理与结果解析
│   │       └── decode/             # 装甲板 YOLO v5/v8、符文 v8 解码
│   ├── settings/                    # 配置类型与 YAML 读取工具
│   ├── tools/                       # 日志工具
│   └── types/                       # 装甲板、符文检测结果类型
├── configs/                         # YAML 配置文件
│   ├── default.yaml
│   └── test_demo.yaml
├── assets/                          # ONNX、OpenVINO、TensorRT 模型资源
│   └── video/                       # 测试视频
├── docs/                           # 部署与使用文档
│   ├── opencv-cuda.md               # OpenCV CUDA 构建、安装与本地验证
│   └── tensorrt.md                  # TensorRT 安装、环境配置与本地验证
├── src/
│   └── test_fps.cpp                 # 视频队列、耗时/FPS 测试与检测可视化
├── README.md
└── LICENSE
```


## 9. 参考项目与致谢

本项目的部署思路、模型输出契约、模型权重与测试视频来自深圳大学 RobotPilots 战队视觉组的开源项目
[SZURPVision/26_NNDeployment_Lib_and_Detection_Models](https://github.com/SZURPVision/26_NNDeployment_Lib_and_Detection_Models)
（NNdeployment：RoboMaster 视觉模型统一部署库），感谢该项目作者与战队视觉组的开源工作。

### 9.1 借鉴内容

| 内容 | 本项目对应位置 |
|---|---|
| 部署层的职责划分：预处理、推理、后处理分离，对外提供统一的检测入口 | `backend/` 整体分层 |
| 装甲板四关键点模型的输出契约：v5 为 `[1, 25200, 22]`（四点、objectness、4 色、9 类），v8 为 `[1, 21, 6300]`（4 色、9 类、四点） | `backend/processing/postprocess/decode/armor_v5.cpp`、`armor_v8.cpp` |
| 装甲板类别 ID 0–8 与颜色 ID 0–3 的定义与顺序 | `backend/types/detection_result.cpp` |
| 能量机关五关键点模型的输出契约与关键点顺序 `top`、`left`、`R`、`right`、`bottom`，以及 `point_R` 命名 | `backend/processing/postprocess/decode/rune_v8.cpp`、`backend/types/detection_result.cpp` |
| 开源模型权重：`Infantry-v5n`、`Infantry-v8n`、`Rune-v8n-fp16` | `assets/armor_v5n_fp16.*`、`armor_v8n_fp16.*`、`rune_v8n_fp16.*` |
| 测试视频：`装甲板.mp4`、`符.avi` 及各距离/姿态陀螺与平移片段 | `assets/video/` |
| 离线视频加 FPS 的性能验证思路 | `src/test_fps.cpp` |

### 9.2 本项目的调整

- 配置由 JSON 改为 YAML 的单一 `pipeline` 节点，按 `PipelineConfig` 分发到 `processing` 与 `inference`，详见第 4 节。
- 对外入口合并为 `Detector` 单个类，提供 `detect()` 与 `submit()`/`collect()`，不再区分装甲板与能量机关两套模型类；两种结果由 `DetectionResults` 变体承载。
- 推理后端由编译期选择改为运行时由 `pipeline.inference.mode`、`type` 指定，同一份构建可切换 TensorRT 与 OpenVINO。
- 预处理提供 CPU 与 CUDA 两种实现，由 `pipeline.processing.preprocess.implementation` 切换；TensorRT 侧新增 engine 同名缓存查找、以 ONNX 为源重构建，以及直接加载 engine 失败时的 ONNX 回退，详见第 5 节。
- 推理队列改为容量与 worker 数可配的异步任务队列，`test_fps` 改为后台解码线程加帧队列，测量口径见第 7 节。

### 9.3 许可证

参考项目采用 MIT License，版权归 `SZURPVision`（2026）所有，并声明其随仓代码、模型、数据、视频、图片和文档均可公开与再分发。
本项目的源码与文档同样以 MIT License 发布，见根目录 `LICENSE`。再分发本项目时，需要同时保留本项目与上游项目的版权声明与许可声明。
OpenCV、OpenVINO、CUDA Toolkit、TensorRT 等运行依赖适用各自许可，不因上述声明而变更。
