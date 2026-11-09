# TensorRT 部署

[返回 README](../README.md)

本教程面向 Ubuntu 22.04 x86_64 和 TensorRT 10.10.0.31，
按“准备环境 → 下载安装包 → 安装 → 配置环境 → 验证”的顺序操作。
安装内容包括 C++ 开发头文件、完整运行库、ONNX 解析器和命令行工具。

## 1. 准备显卡驱动与 CUDA Toolkit

先安装 NVIDIA 驱动和 CUDA Toolkit，可参考
[NVIDIA CUDA Linux 安装指南](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)。

```bash
nvidia-smi
nvcc --version
uname -m
```

确认 GPU 可用，且系统架构为 `x86_64`。`nvcc --version` 用于查看实际 Toolkit 版本；
`nvidia-smi` 中的 CUDA Version 表示驱动支持的版本上限。

TensorRT 10.10 支持 CUDA 11.8，但下载包仍需选择匹配的 CUDA 系列和平台，
不要将 CUDA 12 的安装包用于只有 CUDA 11 依赖的环境。
具体兼容范围见 [TensorRT 10.10 发布说明](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/getting-started/release-notes-10/10.10.0.html)。
Jetson 使用与 JetPack 配套的安装方式，不使用下文的 Ubuntu x86_64 包。

## 2. 下载 TensorRT

进入 [NVIDIA TensorRT 下载页面](https://developer.nvidia.com/tensorrt/download)，
选择 TensorRT 10.10、Ubuntu 22.04、x86_64 以及适配当前 CUDA 环境的安装包。
下载可能需要登录 NVIDIA 账号并接受许可协议。

可以选择以下一种方式：

| 方式 | 用途 |
|---|---|
| Debian local repo（`.deb`） | 通过 APT 安装到系统目录，步骤见第 3 节 |
| Tar（`.tar.gz`） | 解压到用户目录，方便使用独立安装路径，步骤见第 4 节 |

以官方页面实际提供的包名和下载链接为准；不要混用 TensorRT 10.6、8.5.1 等不同版本的包名、仓库目录或密钥路径。

## 3. Debian 本地仓库安装

参考 [NVIDIA Debian 安装指南](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/installing-tensorrt/install-debian.html)。
以下路径需要改为实际下载的文件；`TRT_REPO_PACKAGE` 从包元数据读取，避免手动拼错仓库目录。

```bash
TRT_REPO_DEB="$HOME/下载/实际下载的nv-tensorrt-local-repo安装包.deb"
TRT_REPO_PACKAGE="$(dpkg-deb --field "$TRT_REPO_DEB" Package)"

sudo dpkg -i "$TRT_REPO_DEB"
sudo cp "/var/$TRT_REPO_PACKAGE/"*-keyring.gpg /usr/share/keyrings/
sudo apt update
```

本地仓库包只注册软件源，还需要安装实际的开发库和工具。
先查看可安装版本，选择同一套 TensorRT 10.10 和 CUDA 环境对应的完整版本号：

```bash
apt-cache madison libnvinfer-dev
apt-cache madison libnvonnxparsers-dev
apt-cache madison libnvinfer-bin
```

将选定的完整版本号输入下面的提示。显式指定版本，避免机器上其他 NVIDIA 软件源将依赖升级到另一套版本。

```bash
read -r -p '输入上述三个包共有的完整 TensorRT 10.10 版本号: ' TRT_APT_VERSION
sudo apt install \
  "libnvinfer-dev=$TRT_APT_VERSION" \
  "libnvonnxparsers-dev=$TRT_APT_VERSION" \
  "libnvinfer-bin=$TRT_APT_VERSION"
```

这些开发包会带入所需运行库。也可按官方指南安装 `tensorrt` 元包以获取更完整的组件和 Python 绑定，
仅使用 C++ 接口时不要求 Python 绑定。安装前同样需要核对 APT 选择的版本。

## 4. Tar 独立目录安装（与第 3 节二选一）

下载匹配版本与平台的 tar 包后，修改下面的文件路径再解压。
解压目录名以实际文件为准，本教程示例使用 `TensorRT-10.10.0.31`。

```bash
TRT_ARCHIVE="$HOME/下载/实际下载的TensorRT安装包.tar.gz"
mkdir -p "$HOME/tensorrt"
tar -xzf "$TRT_ARCHIVE" -C "$HOME/tensorrt"

ls "$HOME/tensorrt/TensorRT-10.10.0.31/include/NvInfer.h"
ls "$HOME/tensorrt/TensorRT-10.10.0.31/lib/libnvinfer.so"
ls "$HOME/tensorrt/TensorRT-10.10.0.31/lib/libnvonnxparser.so"
ls "$HOME/tensorrt/TensorRT-10.10.0.31/bin/trtexec"
```

仅使用上述 C++ 头文件、库和工具时，无需再安装 tar 包内的 Python wheel。

## 5. 配置环境并验证

### Debian 安装

检查安装版本及 `trtexec` 所在位置：

```bash
dpkg-query -W 'libnvinfer*' 'libnvonnxparsers*'
dpkg -L libnvinfer-bin | grep '/trtexec$'
```

`trtexec` 常见路径为 `/usr/src/tensorrt/bin/trtexec`，以实际包内容为准。
将其所在目录加入 PATH。以下脚本中的目录需要与上一条命令输出对应：

```bash
mkdir -p "$HOME/opt"
cat > "$HOME/opt/tensorrt_env.sh" <<'ENV'
export PATH="/usr/src/tensorrt/bin:$PATH"
ENV
source "$HOME/opt/tensorrt_env.sh"
```

系统包的运行库通常由动态链接器自动发现，无需另加 tar 安装目录到 `LD_LIBRARY_PATH`。

### Tar 安装

```bash
mkdir -p "$HOME/opt"
cat > "$HOME/opt/tensorrt_env.sh" <<'ENV'
export TENSORRT_ROOT="$HOME/tensorrt/TensorRT-10.10.0.31"
export PATH="$TENSORRT_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$TENSORRT_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
ENV
source "$HOME/opt/tensorrt_env.sh"
```

两种环境脚本选择与安装方式匹配的一种，每次打开新终端时重新 `source`。
`samples` 是示例源码目录，无需加入 PATH；只有可执行工具所在的 `bin` 目录需要加入。

验证工具能启动并检查运行库依赖：

```bash
command -v trtexec
trtexec --help
ldd "$(command -v trtexec)" | grep 'not found'
```

最后一条命令无输出表示未发现缺失的动态库；它不代表 GPU 推理已经验证成功。
如果缺少 CUDA 库，检查下载包的 CUDA 系列、Toolkit 安装与动态库搜索路径。
