# OpenCV CUDA 部署

[返回 README](../README.md)

以下按 Ubuntu 22.04、OpenCV 4.13.0、CUDA Toolkit 11.8 和 RTX 3060 Ti 举例，
将 OpenCV 安装到 `$HOME/opt/opencv-cuda`，与系统 OpenCV 分开使用。

## 1. 准备显卡驱动与 CUDA Toolkit

先参考 [NVIDIA CUDA Toolkit 安装文档](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
完成显卡驱动与 Toolkit 安装，确认以下命令可用：

```bash
nvidia-smi
nvcc --version
nvidia-smi --query-gpu=compute_cap --format=csv
```

最后一条命令输出 GPU 计算能力。例如 `8.6` 对应本例的
`CMAKE_CUDA_ARCHITECTURES=86`；旧方式中的对应参数是 `CUDA_ARCH_BIN=8.6`。
`nvidia-smi` 显示的 CUDA Version 是驱动支持的版本上限，Toolkit 版本以 `nvcc --version` 为准。

## 2. 准备 cuDNN（仅 OpenCV DNN CUDA 需要）

仅使用 OpenCV CUDA 图像处理模块时，可以关闭 `WITH_CUDNN` 和 `OPENCV_DNN_CUDA`，
跳过此步。下文默认采用这组设置。
如果还需要使用 OpenCV 自身的 DNN CUDA 后端，可从
[NVIDIA cuDNN 安装指南](https://docs.nvidia.com/deeplearning/cudnn/installation/latest/index.html)
获取与 CUDA 和平台匹配的版本，也可以从 PyPI 下载 wheel 并解包。
CUDA 11 使用 `nvidia-cudnn-cu11`；CUDA 12 使用 `nvidia-cudnn-cu12`，具体支持范围以所选版本为准。

以下为 CUDA 11 的 wheel 解包方式，需要可用的 Python 3 和 pip；此命令用于下载，配置时仍需显式指定头文件与库路径：

```bash
OPENCV_CUDNN_WHEEL_DIR="$(mktemp -d)"
python3 -m pip download --no-deps nvidia-cudnn-cu11 -d "$OPENCV_CUDNN_WHEEL_DIR"
mkdir -p "$HOME/opt/cudnn"
python3 -m zipfile -e "$OPENCV_CUDNN_WHEEL_DIR"/nvidia_cudnn_cu11-*.whl "$HOME/opt/cudnn"

# wheel 中可能只有带版本号的库文件，为 CMake 查找创建链接。
for lib in "$HOME/opt/cudnn/nvidia/cudnn/lib"/libcudnn.so.*; do
  ln -sf "$(basename "$lib")" "$HOME/opt/cudnn/nvidia/cudnn/lib/libcudnn.so"
done

ls "$HOME/opt/cudnn/nvidia/cudnn/include/cudnn.h"
ls -l "$HOME/opt/cudnn/nvidia/cudnn/lib/libcudnn.so"
```

## 3. 安装编译依赖、视频编解码库与 GUI 库

```bash
sudo apt update
sudo apt install -y build-essential cmake ninja-build git pkg-config \
  libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libavdevice-dev \
  libjpeg-dev libpng-dev libtiff-dev libgtk-3-dev
```

本教程使用 CMake 3.28 或更新版本，运行 `cmake --version` 确认；系统软件源提供的版本不足时，需另行安装。

## 4. 获取源码并配置

OpenCV 与 opencv_contrib 使用相同版本，CUDA 扩展模块位于 opencv_contrib：

```bash
mkdir -p "$HOME/opencv_build"
cd "$HOME/opencv_build"
git clone --depth 1 --branch 4.13.0 https://github.com/opencv/opencv.git
git clone --depth 1 --branch 4.13.0 https://github.com/opencv/opencv_contrib.git
```

将 `OPENCV_CUDA_TOOLKIT` 改为实际安装目录。本机示例为 `$HOME/cuda-11.8`，
其他机器也可能安装在 `/usr/local/cuda` 或 `/usr/local/cuda-11.8`。
路径中使用 `$HOME`，不要写成不会展开的 `"~/..."`。

```bash
OPENCV_CUDA_TOOLKIT="$HOME/cuda-11.8"

cmake -S "$HOME/opencv_build/opencv" -B "$HOME/opencv_build/build_cuda" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$HOME/opt/opencv-cuda" \
  -DCMAKE_INSTALL_LIBDIR=lib \
  -DOPENCV_EXTRA_MODULES_PATH="$HOME/opencv_build/opencv_contrib/modules" \
  -DWITH_CUDA=ON \
  -DENABLE_CUDA_FIRST_CLASS_LANGUAGE=ON \
  -DCMAKE_CUDA_COMPILER="$OPENCV_CUDA_TOOLKIT/bin/nvcc" \
  -DCUDAToolkit_ROOT="$OPENCV_CUDA_TOOLKIT" \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DWITH_CUDNN=OFF \
  -DOPENCV_DNN_CUDA=OFF \
  -DWITH_NVCUVID=OFF \
  -DWITH_NVCUVENC=OFF \
  -DBUILD_opencv_cudacodec=OFF \
  -DOPENCV_GENERATE_PKGCONFIG=ON \
  -DBUILD_SHARED_LIBS=ON \
  -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF \
  -DBUILD_JAVA=OFF -DBUILD_DOCS=OFF
```

若已完成 cuDNN 安装且需要 OpenCV DNN CUDA，将上述 `WITH_CUDNN`、`OPENCV_DNN_CUDA`
改为 `ON`，并在同一配置命令中追加以下参数：

```bash
  -DCUDNN_INCLUDE_DIR="$HOME/opt/cudnn/nvidia/cudnn/include" \
  -DCUDNN_LIBRARY="$HOME/opt/cudnn/nvidia/cudnn/lib/libcudnn.so"
```

关键选项说明：

| 选项 | 作用 |
|---|---|
| `CMAKE_INSTALL_PREFIX` | 使用独立安装目录，避免覆盖系统 OpenCV |
| `OPENCV_EXTRA_MODULES_PATH` | 指向匹配版本的 opencv_contrib 模块目录 |
| `WITH_CUDA=ON` | 启用 OpenCV CUDA 模块 |
| `ENABLE_CUDA_FIRST_CLASS_LANGUAGE=ON` | 使用原生 CUDA 语言支持及 `FindCUDAToolkit`，适配 CMake 3.28 |
| `CMAKE_CUDA_ARCHITECTURES=86` | 选择目标 GPU 架构，示例对应计算能力 8.6 |
| `CUDAToolkit_ROOT` / `CMAKE_CUDA_COMPILER` | 指定 Toolkit 与其中的 nvcc，避免选错安装版本 |
| `WITH_CUDNN` / `OPENCV_DNN_CUDA` | 启用 OpenCV DNN CUDA 后端，CUDA 图像处理模块不需要 |
| `BUILD_opencv_cudacodec=OFF` | 本例不构建依赖 NVIDIA Video Codec SDK 的编解码模块 |
| `OPENCV_GENERATE_PKGCONFIG=ON` | 生成供 pkg-config 使用的配置文件 |

原生 CUDA 配置依据见 [OpenCV CUDA 检测逻辑](https://github.com/opencv/opencv/blob/4.13.0/cmake/OpenCVDetectCUDALanguage.cmake)
和 [OpenCV 配置生成逻辑](https://github.com/opencv/opencv/blob/4.13.0/cmake/OpenCVGenConfig.cmake)。

## 5. 检查配置摘要，编译并安装

配置摘要应包含：

```text
NVIDIA CUDA: YES
To be built: ... cudaarithm ... cudawarping ...
GUI: ... GTK ...
Video I/O:
  FFMPEG: YES
```

这些内容分布在摘要的不同位置，格式可能随环境变化。如果启用了 cuDNN，还应看到 `cuDNN: YES`。
若 CUDA 未启用，检查 Toolkit 路径、nvcc、宿主编译器和配置阶段报错；若缺少 GUI 或 FFmpeg，先补齐对应依赖再重新配置。

```bash
cmake --build "$HOME/opencv_build/build_cuda" --parallel 4
cmake --install "$HOME/opencv_build/build_cuda"
```

安装到当前用户目录无需 `sudo`。内存不足时降低并行数；显式指定构建目录，避免误在源码目录中执行安装。

## 6. 创建环境脚本并验证本地安装

```bash
cat > "$HOME/opt/opencv-cuda/opencv_cuda_env.sh" <<'ENV'
export OpenCV_DIR="$HOME/opt/opencv-cuda/lib/cmake/opencv4"
export PKG_CONFIG_PATH="$HOME/opt/opencv-cuda/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
export LD_LIBRARY_PATH="$HOME/opt/opencv-cuda/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# 若安装并启用了 cuDNN，取消下一行注释。
# export LD_LIBRARY_PATH="$HOME/opt/cudnn/nvidia/cudnn/lib:$LD_LIBRARY_PATH"
ENV

source "$HOME/opt/opencv-cuda/opencv_cuda_env.sh"
pkg-config --modversion opencv4
```

使用新终端时重新 `source`。`pkg-config --modversion opencv4` 应输出 `4.13.0`。
还可以检查安装目录中的配置、模块库及动态库依赖：

```bash
ls "$OpenCV_DIR/OpenCVConfig.cmake"
ls "$HOME/opt/opencv-cuda/lib/libopencv_cudaarithm.so"
ls "$HOME/opt/opencv-cuda/lib/libopencv_cudawarping.so"
ldd "$HOME/opt/opencv-cuda/lib/libopencv_cudawarping.so" | grep 'not found'
```

最后一条命令无输出表示未发现缺失的动态库。结合构建时的 CUDA 配置摘要，
确认版本、CUDA 模块和依赖库均符合预期，即完成本地安装检查；这些检查尚未执行实际 GPU 图像处理。
