cmake_minimum_required(VERSION 3.22)
set(lib_name backend)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ============================================================================
# 依赖开关与路径：每个依赖“检测到即编译”，可用后端全部编进同一个库，
# 运行时由 YAML 的 deploy_way / preprocess_cuda / postprocess_cuda 选择，
# 切换后端或 CUDA 只需改配置，不必重新编译。
#   -DNNDEPLOYMENT_OPENCV_CUDA_DIR=<带CUDA的OpenCV cmake目录>  （也认同名环境变量）
#   -DOpenCV_DIR=<OpenCV cmake 目录>  -DOPENVINO_DIR=<runtime/cmake>
#   -DTENSORRT_ROOT=<TensorRT 根目录>
# ============================================================================
set(OPENVINO_DIR "${OPENVINO_DIR}" CACHE PATH "OpenVINO runtime 的 cmake 目录（留空则用系统/环境变量）")
set(TENSORRT_ROOT "" CACHE PATH "TensorRT 安装根目录（含 include/ 与 lib/）")
set(NNDEPLOYMENT_OPENCV_CUDA_DIR "${NNDEPLOYMENT_OPENCV_CUDA_DIR}" CACHE PATH "带 CUDA 模块的 OpenCV cmake 目录（优先使用）")
option(NNDEPLOYMENT_ENABLE_OPENVINO "检测到 OpenVINO 时编译该后端" ON)
option(NNDEPLOYMENT_ENABLE_TENSORRT "检测到 TensorRT 时编译该后端" ON)
option(NNDEPLOYMENT_ENABLE_OPENCV_CUDA "检测到 OpenCV CUDA 模块时启用预处理/后处理加速" ON)

# ---------------------------------------------------------------------------
# 选择带 CUDA 的 OpenCV（必须在 find_package(OpenCV) 之前）
# 坑：build/CMakeCache.txt 里残留的 OpenCV_DIR 可能指向无 CUDA 的系统 OpenCV
# （如 /usr/lib/cmake/opencv4）。这里按候选目录探测并纠正旧缓存。判断方式是
# 直接读 OpenCVModules.cmake —— 二次 find_package 会因导入目标重复而报
# “Some but not all targets ... already defined”。
# ---------------------------------------------------------------------------
function(nn_opencv_dir_has_cuda dir out_var)
    set(${out_var} FALSE PARENT_SCOPE)
    if(NOT EXISTS "${dir}/OpenCVModules.cmake")
        return()
    endif()
    file(READ "${dir}/OpenCVModules.cmake" _nn_mods)
    if(_nn_mods MATCHES "opencv_cudawarping" AND _nn_mods MATCHES "opencv_cudaarithm")
        set(${out_var} TRUE PARENT_SCOPE)
    endif()
endfunction()

if(NNDEPLOYMENT_ENABLE_OPENCV_CUDA)
    set(_nn_cv_cuda_dir "")
    foreach(_nn_dir
        "$ENV{NNDEPLOYMENT_OPENCV_CUDA_DIR}" "${NNDEPLOYMENT_OPENCV_CUDA_DIR}" "${OpenCV_DIR}"
        /usr/local/lib/cmake/opencv4 /usr/local/lib/cmake/OpenCV /usr/lib/cmake/opencv4
        /usr/lib/aarch64-linux-gnu/cmake/opencv4 /usr/lib/x86_64-linux-gnu/cmake/opencv4
    )
        nn_opencv_dir_has_cuda("${_nn_dir}" _nn_has)
        if(_nn_dir AND _nn_has AND NOT _nn_cv_cuda_dir)
            set(_nn_cv_cuda_dir "${_nn_dir}")
        endif()
    endforeach()

    if(_nn_cv_cuda_dir)
        # 显式指定的 OpenCV_DIR（含旧缓存）不含 CUDA 时，警告并改用带 CUDA 的那份。
        nn_opencv_dir_has_cuda("${OpenCV_DIR}" _nn_cur_has)
        if(OpenCV_DIR AND NOT OpenCV_DIR STREQUAL _nn_cv_cuda_dir AND NOT _nn_cur_has)
            message(WARNING "NNdeployment: 指定的 OpenCV_DIR=${OpenCV_DIR} 不含 CUDA 模块，自动改用 ${_nn_cv_cuda_dir}；如需强制使用指定版本，请加 -DNNDEPLOYMENT_ENABLE_OPENCV_CUDA=OFF")
        endif()
        set(OpenCV_DIR "${_nn_cv_cuda_dir}" CACHE PATH "OpenCV 配置目录（优先带 CUDA 模块）" FORCE)
        message(STATUS "NNdeployment: 选用带 CUDA 模块的 OpenCV：${_nn_cv_cuda_dir}")
    else()
        message(STATUS "NNdeployment: 未找到带 CUDA 模块的 OpenCV（可用 -DNNDEPLOYMENT_OPENCV_CUDA_DIR=<...> 指定）")
    endif()
endif()

# OpenCV 必需；模型配置用 YAML（与 qianli_vision 一致）；CUDA Toolkit 可选。
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(CUDAToolkit QUIET)

# nvcc 不在 PATH 时（远程命令 / IDE / CI 的非登录 shell 常见）用 CUDAToolkit 的
# nvcc 兜底，否则 enable_language(CUDA) 失败。
if(CUDAToolkit_NVCC_EXECUTABLE AND EXISTS "${CUDAToolkit_NVCC_EXECUTABLE}")
    find_program(_nn_nvcc_in_path nvcc)
    if(NOT _nn_nvcc_in_path)
        message(STATUS "NNdeployment: nvcc 不在 PATH，改用 CUDAToolkit 提供的 ${CUDAToolkit_NVCC_EXECUTABLE}")
        set(CMAKE_CUDA_COMPILER "${CUDAToolkit_NVCC_EXECUTABLE}" CACHE FILEPATH "CUDA 编译器" FORCE)
    endif()
endif()

# ---------------------------------------------------------------------------
# OpenVINO 后端
# ---------------------------------------------------------------------------
set(NNDEPLOYMENT_WITH_OPENVINO OFF)
if(NNDEPLOYMENT_ENABLE_OPENVINO)
    find_package(OpenVINO QUIET COMPONENTS Runtime)
    if(OpenVINO_FOUND)
        set(NNDEPLOYMENT_WITH_OPENVINO ON)
        message(STATUS "NNdeployment: 启用 OpenVINO 后端 (${OpenVINO_DIR})")
    else()
        message(STATUS "NNdeployment: 未找到 OpenVINO，跳过 OpenVINO 后端")
    endif()
endif()

# ---------------------------------------------------------------------------
# TensorRT 后端：优先用官方 CMake package，找不到时按 TENSORRT_ROOT 与常见
# 安装根目录用 find_path/find_library 直接查找（PATH_SUFFIXES 覆盖 lib/lib64）。
# ---------------------------------------------------------------------------
set(NNDEPLOYMENT_WITH_TENSORRT OFF)
if(NNDEPLOYMENT_ENABLE_TENSORRT AND CUDAToolkit_FOUND)
    find_package(TensorRT QUIET)
    if(NOT TensorRT_FOUND)
        set(_nn_trt_roots ${TENSORRT_ROOT} /usr/local/tensorrt /usr/src/tensorrt)
        find_path(TENSORRT_INCLUDE_DIR NAMES NvInfer.h HINTS ${_nn_trt_roots} PATH_SUFFIXES include)
        find_library(TENSORRT_NVINFER_LIB NAMES nvinfer HINTS ${_nn_trt_roots} PATH_SUFFIXES lib lib64)
        find_library(TENSORRT_NVONNXPARSER_LIB NAMES nvonnxparser HINTS ${_nn_trt_roots} PATH_SUFFIXES lib lib64)
        if(TENSORRT_INCLUDE_DIR AND TENSORRT_NVINFER_LIB AND TENSORRT_NVONNXPARSER_LIB)
            set(TensorRT_FOUND TRUE)
            set(TENSORRT_INCLUDE_DIRS ${TENSORRT_INCLUDE_DIR})
            set(TENSORRT_LIBRARIES ${TENSORRT_NVINFER_LIB} ${TENSORRT_NVONNXPARSER_LIB})
        endif()
    endif()
    if(TensorRT_FOUND)
        set(NNDEPLOYMENT_WITH_TENSORRT ON)
        message(STATUS "NNdeployment: 启用 TensorRT 后端 (${TENSORRT_INCLUDE_DIR})")
    else()
        message(STATUS "NNdeployment: 未找到 TensorRT，跳过 TensorRT 后端")
    endif()
endif()

# ---------------------------------------------------------------------------
# OpenCV CUDA 加速（预处理/后处理）：检查 OpenCV 是否提供 CUDA 模块目标即可，
# 未启用时按原因给出对应的修复提示。
# ---------------------------------------------------------------------------
set(NNDEPLOYMENT_WITH_OPENCV_CUDA OFF)
if(NNDEPLOYMENT_ENABLE_OPENCV_CUDA AND CUDAToolkit_FOUND AND TARGET opencv_cudawarping AND TARGET opencv_cudaarithm)
    enable_language(CUDA)
    set(NNDEPLOYMENT_WITH_OPENCV_CUDA ON)
    message(STATUS "NNdeployment: 启用 OpenCV CUDA 预处理/后处理 (OpenCV ${OpenCV_VERSION} @ ${OpenCV_DIR})")
elseif(NOT NNDEPLOYMENT_ENABLE_OPENCV_CUDA)
    message(STATUS "NNdeployment: OpenCV CUDA 被手动关闭（-DNNDEPLOYMENT_ENABLE_OPENCV_CUDA=OFF）")
elseif(NOT CUDAToolkit_FOUND)
    message(STATUS "NNdeployment: 未启用 OpenCV CUDA —— 未找到 CUDA Toolkit（安装 CUDA，或用 -DCUDAToolkit_ROOT=<...> 指定）")
else()
    message(STATUS "NNdeployment: 未启用 OpenCV CUDA —— 当前 OpenCV 无 CUDA 模块（用 -DNNDEPLOYMENT_OPENCV_CUDA_DIR=<带 CUDA 的 OpenCV cmake 目录> 指定；若旧缓存把 OpenCV_DIR 固定在无 CUDA 版本，也可直接删掉 build/ 重配）")
endif()

if(NOT NNDEPLOYMENT_WITH_OPENVINO AND NOT NNDEPLOYMENT_WITH_TENSORRT)
    message(FATAL_ERROR "未找到任何推理后端（OpenVINO 或 TensorRT），无法构建 NNdeployment_lib")
endif()

# 供 app_plugin 等子目录判断库中实际编译了哪些能力。
foreach(_nn_cap OPENVINO TENSORRT OPENCV_CUDA)
    set(NNDEPLOYMENT_WITH_${_nn_cap} "${NNDEPLOYMENT_WITH_${_nn_cap}}" CACHE INTERNAL "NNdeployment: ${_nn_cap} 后端已编译")
endforeach()