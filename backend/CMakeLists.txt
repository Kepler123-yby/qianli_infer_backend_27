cmake_minimum_required(VERSION 3.22)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(backend LANGUAGES CXX)
endif()

set(lib_name backend)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ============================================================================
# 依赖开关与路径：每个依赖“检测到即编译”，可用后端全部编进同一个库，
# 运行时由 YAML 的 pipeline.inference.type / pipeline.processing.preprocess.implementation 选择，
# 切换后端或预处理 CUDA 只需改配置，不必重新编译；后处理固定使用 CPU。
#   -DQIANLI_OPENCV_CUDA_DIR=<带CUDA的OpenCV cmake目录>  （也认同名环境变量）
#   -DOpenCV_DIR=<OpenCV cmake 目录>  -DOPENVINO_DIR=<runtime/cmake>
#   -DTENSORRT_ROOT=<TensorRT 根目录>
# ============================================================================
set(OPENVINO_DIR "${OPENVINO_DIR}" CACHE PATH "OpenVINO runtime 的 cmake 目录（留空则用系统/环境变量）")
set(TENSORRT_ROOT "" CACHE PATH "TensorRT 安装根目录（含 include/ 与 lib/）")
set(QIANLI_OPENCV_CUDA_DIR "${QIANLI_OPENCV_CUDA_DIR}" CACHE PATH "带 CUDA 模块的 OpenCV cmake 目录（优先使用）")
option(QIANLI_ENABLE_OPENVINO "检测到 OpenVINO 时编译该后端" ON)
option(QIANLI_ENABLE_TENSORRT "检测到 TensorRT 时编译该后端" ON)
option(QIANLI_ENABLE_OPENCV_CUDA "检测到 OpenCV CUDA 模块时启用预处理加速" ON)

# ---------------------------------------------------------------------------
# 选择带 CUDA 的 OpenCV（必须在 find_package(OpenCV) 之前）
# 坑：build/CMakeCache.txt 里残留的 OpenCV_DIR 可能指向无 CUDA 的系统 OpenCV
# （如 /usr/lib/cmake/opencv4）。这里按候选目录探测并纠正旧缓存。判断方式是
# 直接读 OpenCVModules.cmake —— 二次 find_package 会因导入目标重复而报
# “Some but not all targets ... already defined”。
# ---------------------------------------------------------------------------
function(qianli_opencv_dir_has_cuda dir out_var)
    set(${out_var} FALSE PARENT_SCOPE)
    if(NOT EXISTS "${dir}/OpenCVModules.cmake")
        return()
    endif()
    file(READ "${dir}/OpenCVModules.cmake" _qianli_opencv_modules)
    if(_qianli_opencv_modules MATCHES "opencv_cudawarping" AND _qianli_opencv_modules MATCHES "opencv_cudaarithm")
        set(${out_var} TRUE PARENT_SCOPE)
    endif()
endfunction()

if(QIANLI_ENABLE_OPENCV_CUDA)
    set(_qianli_opencv_cuda_dir "")
    foreach(_qianli_opencv_candidate_dir
        "$ENV{QIANLI_OPENCV_CUDA_DIR}" "${QIANLI_OPENCV_CUDA_DIR}" "${OpenCV_DIR}"
        /usr/local/lib/cmake/opencv4 /usr/local/lib/cmake/OpenCV /usr/lib/cmake/opencv4
        /usr/lib/aarch64-linux-gnu/cmake/opencv4 /usr/lib/x86_64-linux-gnu/cmake/opencv4
    )
        qianli_opencv_dir_has_cuda("${_qianli_opencv_candidate_dir}" _qianli_opencv_has_cuda)
        if(_qianli_opencv_candidate_dir AND _qianli_opencv_has_cuda AND NOT _qianli_opencv_cuda_dir)
            set(_qianli_opencv_cuda_dir "${_qianli_opencv_candidate_dir}")
        endif()
    endforeach()

    if(_qianli_opencv_cuda_dir)
        # 显式指定的 OpenCV_DIR（含旧缓存）不含 CUDA 时，警告并改用带 CUDA 的那份。
        qianli_opencv_dir_has_cuda("${OpenCV_DIR}" _qianli_opencv_current_has_cuda)
        if(OpenCV_DIR AND NOT OpenCV_DIR STREQUAL _qianli_opencv_cuda_dir AND NOT _qianli_opencv_current_has_cuda)
            message(WARNING "QIANLI: 指定的 OpenCV_DIR=${OpenCV_DIR} 不含 CUDA 模块，自动改用 ${_qianli_opencv_cuda_dir}；如需强制使用指定版本，请加 -DQIANLI_ENABLE_OPENCV_CUDA=OFF")
        endif()
        set(OpenCV_DIR "${_qianli_opencv_cuda_dir}" CACHE PATH "OpenCV 配置目录（优先带 CUDA 模块）" FORCE)
        message(STATUS "QIANLI: 选用带 CUDA 模块的 OpenCV：${_qianli_opencv_cuda_dir}")
    else()
        message(STATUS "QIANLI: 未找到带 CUDA 模块的 OpenCV（可用 -DQIANLI_OPENCV_CUDA_DIR=<...> 指定）")
    endif()
endif()

# OpenCV 必需；模型配置用 YAML（与 qianli_vision 一致）；CUDA Toolkit 可选。
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(spdlog REQUIRED)
find_package(Threads REQUIRED)
find_package(CUDAToolkit QUIET)

# nvcc 不在 PATH 时（远程命令 / IDE / CI 的非登录 shell 常见）用 CUDAToolkit 的
# nvcc 兜底，否则 enable_language(CUDA) 失败。
if(CUDAToolkit_NVCC_EXECUTABLE AND EXISTS "${CUDAToolkit_NVCC_EXECUTABLE}")
    find_program(_qianli_nvcc_in_path nvcc)
    if(NOT _qianli_nvcc_in_path)
        message(STATUS "QIANLI: nvcc 不在 PATH，改用 CUDAToolkit 提供的 ${CUDAToolkit_NVCC_EXECUTABLE}")
        set(CMAKE_CUDA_COMPILER "${CUDAToolkit_NVCC_EXECUTABLE}" CACHE FILEPATH "CUDA 编译器" FORCE)
    endif()
endif()

# ---------------------------------------------------------------------------
# OpenVINO 后端
# ---------------------------------------------------------------------------
set(QIANLI_WITH_OPENVINO OFF)
if(QIANLI_ENABLE_OPENVINO)
    find_package(OpenVINO QUIET COMPONENTS Runtime)
    if(OpenVINO_FOUND)
        set(QIANLI_WITH_OPENVINO ON)
        message(STATUS "QIANLI: 启用 OpenVINO 后端 (${OpenVINO_DIR})")
    else()
        message(STATUS "QIANLI: 未找到 OpenVINO，跳过 OpenVINO 后端")
    endif()
endif()

# ---------------------------------------------------------------------------
# TensorRT 后端：优先用官方 CMake package，找不到时按 TENSORRT_ROOT 与常见
# 安装根目录用 find_path/find_library 直接查找（PATH_SUFFIXES 覆盖 lib/lib64）。
# ---------------------------------------------------------------------------
set(QIANLI_WITH_TENSORRT OFF)
if(QIANLI_ENABLE_TENSORRT AND CUDAToolkit_FOUND)
    find_package(TensorRT QUIET)
    if(NOT TensorRT_FOUND)
        set(_qianli_tensorrt_roots ${TENSORRT_ROOT} /usr/local/tensorrt /usr/src/tensorrt)
        find_path(TENSORRT_INCLUDE_DIR NAMES NvInfer.h HINTS ${_qianli_tensorrt_roots} PATH_SUFFIXES include)
        find_library(TENSORRT_NVINFER_LIB NAMES nvinfer HINTS ${_qianli_tensorrt_roots} PATH_SUFFIXES lib lib64)
        find_library(TENSORRT_NVONNXPARSER_LIB NAMES nvonnxparser HINTS ${_qianli_tensorrt_roots} PATH_SUFFIXES lib lib64)
        if(TENSORRT_INCLUDE_DIR AND TENSORRT_NVINFER_LIB AND TENSORRT_NVONNXPARSER_LIB)
            set(TensorRT_FOUND TRUE)
            set(TENSORRT_INCLUDE_DIRS ${TENSORRT_INCLUDE_DIR})
            set(TENSORRT_LIBRARIES ${TENSORRT_NVINFER_LIB} ${TENSORRT_NVONNXPARSER_LIB})
        endif()
    endif()
    if(TensorRT_FOUND)
        set(QIANLI_WITH_TENSORRT ON)
        message(STATUS "QIANLI: 启用 TensorRT 后端 (${TENSORRT_INCLUDE_DIR})")
    else()
        message(STATUS "QIANLI: 未找到 TensorRT，跳过 TensorRT 后端")
    endif()
endif()

# ---------------------------------------------------------------------------
# OpenCV CUDA 加速（预处理）：检查 OpenCV 是否提供 CUDA 模块目标即可，
# 未启用时按原因给出对应的修复提示。
# ---------------------------------------------------------------------------
set(QIANLI_WITH_OPENCV_CUDA OFF)
if(QIANLI_ENABLE_OPENCV_CUDA AND CUDAToolkit_FOUND AND TARGET opencv_cudawarping AND TARGET opencv_cudaarithm)
    enable_language(CUDA)
    set(QIANLI_WITH_OPENCV_CUDA ON)
    message(STATUS "QIANLI: 启用 OpenCV CUDA 预处理 (OpenCV ${OpenCV_VERSION} @ ${OpenCV_DIR})")
elseif(NOT QIANLI_ENABLE_OPENCV_CUDA)
    message(STATUS "QIANLI: OpenCV CUDA 被手动关闭（-DQIANLI_ENABLE_OPENCV_CUDA=OFF）")
elseif(NOT CUDAToolkit_FOUND)
    message(STATUS "QIANLI: 未启用 OpenCV CUDA —— 未找到 CUDA Toolkit（安装 CUDA，或用 -DCUDAToolkit_ROOT=<...> 指定）")
else()
    message(STATUS "QIANLI: 未启用 OpenCV CUDA —— 当前 OpenCV 无 CUDA 模块（用 -DQIANLI_OPENCV_CUDA_DIR=<带 CUDA 的 OpenCV cmake 目录> 指定；若旧缓存把 OpenCV_DIR 固定在无 CUDA 版本，也可直接删掉 build/ 重配）")
endif()

if(NOT QIANLI_WITH_OPENVINO AND NOT QIANLI_WITH_TENSORRT)
    message(FATAL_ERROR "QIANLI: 未找到任何推理后端（OpenVINO 或 TensorRT），无法构建 ${lib_name}")
endif()

# 供上层判断库中实际编译了哪些能力。
foreach(_qianli_capability OPENVINO TENSORRT OPENCV_CUDA)
    set(QIANLI_WITH_${_qianli_capability} "${QIANLI_WITH_${_qianli_capability}}" CACHE INTERNAL "QIANLI: ${_qianli_capability} 后端已编译")
endforeach()

# 未指定库类型，遵循 BUILD_SHARED_LIBS（默认生成静态库）。
# decode/*.cpp、types/detection_result.cpp 和 settings/yaml_utils.cpp 由其他
# 源文件直接包含，不作为独立编译单元重复加入。
add_library(${lib_name}
    detector.cpp
    inference/inference.cpp
    processing/processor.cpp
    processing/preprocess/preprocess.cpp
    processing/postprocess/postprocess.cpp
    settings/pipeline_setting.cpp
    settings/processing_setting.cpp
    tools/logger.cpp
)

target_compile_features(${lib_name} PUBLIC cxx_std_20)
target_include_directories(${lib_name}
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}"
        ${OpenCV_INCLUDE_DIRS}
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/inference"
)
target_link_libraries(${lib_name}
    PUBLIC
        ${OpenCV_LIBS}
        yaml-cpp
        spdlog::spdlog
    PRIVATE
        Threads::Threads
)

if(QIANLI_WITH_OPENVINO)
    target_sources(${lib_name} PRIVATE inference/openvino/openvino_backend.cpp)
    target_link_libraries(${lib_name} PRIVATE openvino::runtime)
endif()

if(QIANLI_WITH_TENSORRT)
    target_sources(${lib_name} PRIVATE inference/tensorrt/tensorrt_backend.cpp)
    target_include_directories(${lib_name} PRIVATE ${TENSORRT_INCLUDE_DIRS})
    target_link_libraries(${lib_name} PRIVATE ${TENSORRT_LIBRARIES} CUDA::cudart)
endif()

if(QIANLI_WITH_OPENCV_CUDA)
    target_link_libraries(${lib_name} PRIVATE opencv_cudaarithm opencv_cudawarping CUDA::cudart)
endif()
