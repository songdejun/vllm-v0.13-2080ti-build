# vLLM v0.13 RTX 2080 Ti 本地编译说明

这个 fork 的目标是在 RTX 2080 Ti 上把 vLLM v0.13 本地编译通过。2080 Ti 是 Turing 架构，CUDA compute capability 为 7.5，不支持 Ampere 以后才完整可用的 bf16 device intrinsic。因此这个 fork 的核心改动不是“让 2080 Ti 支持 bf16”，而是让 sm75 编译时跳过或拒绝这些 bf16 CUDA 路径。

## 测试环境

- GPU: NVIDIA RTX 2080 Ti, sm75
- CUDA: 12.0
- Compiler: gcc-11 / g++-11
- Python: 3.12
- PyTorch: 2.9.0

## 必要环境变量

本地编译时建议固定 host compiler，避免 nvcc 选择到不兼容的 gcc/g++：

```bash
export CC=gcc-11
export CXX=g++-11
export CUDAHOSTCXX=g++-11
export TORCH_CUDA_ARCH_LIST="7.5"
```

用于本地反复调试的可选变量：

```bash
export VLLM_BUILD_TEMP="$PWD/.build-temp"
export CCACHE_BASEDIR="$PWD"
export CCACHE_NOHASHDIR=true
export MAX_JOBS=1
```

说明：

- `VLLM_BUILD_TEMP` 用来固定 CMake 临时构建目录，避免每次 `pip install -e .` 都换一个 `build/temp.*` 目录。
- `CCACHE_BASEDIR` 和 `CCACHE_NOHASHDIR=true` 用来提升 ccache 在固定源码目录下的命中率。
- `MAX_JOBS=1` 不是必须项，主要方便定位编译错误；机器资源足够时可以调大。

## 编译命令

推荐在已经准备好本地 Python 构建依赖后使用：

```bash
MAX_JOBS=1 pip install -e . --no-build-isolation -vvv
```

如果不用 `--no-build-isolation`，pip 会创建临时 build isolation overlay，并在里面安装构建依赖。这是 Python 包构建的正常行为，但会导致临时路径变化，对 CMake 缓存和 ccache 不友好。

## 必须修改的代码

这些改动是 RTX 2080 Ti / sm75 编译通过的关键。

### csrc/quantization/activation_kernels.cu

原始代码里存在 bf16 CUDA intrinsic，例如 `make_bfloat162`、`__bfloat1622float2`，以及 `__nv_bfloat16` 到 `__half` / `__half2` 的隐式路径。sm75 编译设备代码时不支持这些 bf16 intrinsic，因此会直接编译失败。

本 fork 做了这些处理：

- 对 bf16 helper 和 bf16 device overload 增加 `__CUDA_ARCH__ >= 800` 条件。
- 将 CUDA 路径里的 `make_bfloat162(...)` 改为 `__halves2bfloat162(...)`。
- 对 `silu_mul_fp8_quant_deep_gemm_kernel` 的实际 kernel body 增加 sm80+ 编译保护。
- 在运行时对 sm75 上的 bf16 调用给出明确 `TORCH_CHECK` 错误，提示使用 float16。

### csrc/moe/topk_softmax_kernels.cu

这个文件的问题比较隐蔽：即使模板实例化使用的是 `InputType=float`，nvcc 仍会解析 `if constexpr` 分支中的非依赖 bf16 intrinsic 名称，导致 sm75 编译失败。

本 fork 做了这些处理：

- 对 `InputType == __nv_bfloat16` 分支增加 `__CUDA_ARCH__ >= 800` 条件。
- 在运行时拒绝 sm75 上的 bf16 `topk_softmax` 调用，提示使用 float16。

## 构建便利修改

这些改动不是 sm75 编译通过的核心，但能让本地反复编译更稳定。

### setup.py

- 新增 `VLLM_BUILD_TEMP` 支持，用于固定 CMake 构建目录。
- 在 CMake configure 时清理 `Torch_DIR`、`Caffe2_DIR`、`c10_LIBRARY` 等缓存项，避免旧的 pip overlay 临时路径残留在 `CMakeCache.txt` 中。
- 显式传入当前环境里的 `ninja` 路径，避免 CMake 复用已经失效的临时 overlay ninja。

### CMake FetchContent 本地依赖复用

vLLM 的第三方依赖由 CMake `FetchContent` 管理，源码默认放在 `.deps/*-src` 下，stamp 状态放在 `.deps/*-subbuild` 下。正常情况下，只要 `.deps` 中源码目录和 stamp 都健康，CMake 不会重新 clone。

本 fork 增加了若干本地源码自动复用逻辑：当没有设置对应环境变量，但 `.deps/*-src` 已存在时，直接把它当作 `SOURCE_DIR` 使用。涉及：

- `VLLM_CUTLASS_SRC_DIR` / `.deps/cutlass-src`
- `FLASH_MLA_SRC_DIR` / `.deps/flashmla-src`
- `QUTLASS_SRC_DIR` / `.deps/qutlass-src`
- `TRITON_KERNELS_SRC_DIR` / `.deps/triton_kernels-src`
- `VLLM_FLASH_ATTN_SRC_DIR` / `.deps/vllm-flash-attn-src`

这部分属于“本地离线/弱网构建保险”。如果 FetchContent stamp 完整，它们不是必须的；如果 stamp 失效或源码目录被误删，CMake 仍可能尝试下载。

## 运行限制

2080 Ti 不支持原生 bf16。这个 fork 只是修复编译，并在不支持的 bf16 CUDA 路径上提前报错。实际运行模型时建议显式使用 fp16，例如：

```bash
vllm serve <model> --dtype=half
```

不要期望 bf16 模型路径在 2080 Ti 上可用。

## 不应提交到 GitHub 的目录

这些目录是本地构建产物或本地依赖缓存，不应放进仓库：

- `.build-temp/`
- `.deps/`
- `build/`
- `dist/`
- `*.egg-info/`

