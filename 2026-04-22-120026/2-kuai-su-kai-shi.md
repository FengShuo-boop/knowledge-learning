本文档面向初次接触 **TensorFlow MUSA Extension** 的开发者，目标是在 **5 分钟内**完成从源码获取、插件编译到功能验证的完整闭环。阅读前请确保您已具备 Linux 基础操作与 Python 环境管理经验；若尚未安装 MUSA SDK 或 TensorFlow，请先前往 [环境依赖与前置准备](3-huan-jing-yi-lai-yu-qian-zhi-zhun-bei) 完成系统级配置。

Sources: [README.md](README.md#L1-L14)

---

## 前置条件检查

在运行任何命令前，请确认以下软硬件条件已满足。下表列出了构建与运行所需的最低版本及验证命令。

| 类别 | 组件 | 最低版本 | 快速验证命令 |
|------|------|----------|--------------|
| **构建工具** | CMake | >= 3.10 | `cmake --version` |
| | GCC / G++ | 兼容 C++14 | `g++ --version` |
| **MUSA SDK** | MUSA Runtime | >= 1.0 | `ls /usr/local/musa` |
| | muBLAS | 随 SDK 安装 | `ls /usr/local/musa/lib \| grep mublas` |
| | muDNN | 随 SDK 安装 | `ls /usr/local/musa/lib \| grep mudnn` |
| **Python 环境** | Python | >= 3.7 | `python3 --version` |
| | TensorFlow | == 2.6.1 | `python3 -c "import tensorflow as tf; print(tf.__version__)"` |
| | protobuf | == 3.20.3 | `python3 -c "import google.protobuf; print(google.protobuf.__version__)"` |
| | NumPy | >= 1.19.0 | `python3 -c "import numpy; print(numpy.__version__)"` |
| | prettytable | >= 3.0.0 | `python3 -c "import prettytable; print(prettytable.__version__)"` |

如果 MUSA SDK 未安装在默认路径 `/usr/local/musa`，后续构建时可通过修改 `CMakeLists.txt` 中 `MUSA_PATH` 变量指向自定义目录。TensorFlow 版本必须严格匹配 `2.6.1`，因为插件通过 TensorFlow 的 ABI 符号进行内核注册与设备交互，任何版本偏差都会导致运行时链接错误或段错误。

Sources: [README.md](README.md#L36-L55), [CMakeLists.txt](CMakeLists.txt#L1-L22)

---

## 项目结构速览

为了在阅读源码和定位问题时建立空间感，下图展示了与快速开始直接相关的目录布局。核心源码位于 `musa_ext/` 下，构建产物输出到 `build/`，测试用例集中在 `test/` 中。

```text
tensorflow_musa_extension/
├── build.sh                # 一键构建脚本（入口）
├── CMakeLists.txt          # CMake 配置
├── musa_ext/               # 核心源码
│   ├── kernels/            # MUSA Kernel 实现（.cc + .mu）
│   ├── mu/                 # 设备注册、图优化器
│   └── utils/              # 公共工具与日志
├── test/                   # 测试套件
│   ├── test_runner.py      # 统一测试运行器
│   ├── musa_test_utils.py  # 测试基类（自动加载插件）
│   ├── ops/                # 单算子功能测试
│   └── fusion/             # 端到端融合测试
└── docs/
    └── DEBUG_GUIDE.md      # 调试与诊断参考
```

上述结构体现了“构建-加载-测试”的三段式工作流：编译系统从 `musa_ext/` 收集 `.cc` 主机代码与 `.mu` 设备内核，生成单一动态库 `libmusa_plugin.so`；测试框架在导入时自动加载该库，并在 MUSA 设备上执行算子验证。

Sources: [README.md](README.md#L16-L34)

---

## 构建插件

TensorFlow MUSA Extension 使用 **CMake + Make** 作为底层构建系统，并通过 `build.sh` 脚本对外暴露简化的命令行接口。整个编译流程可概括为：解析构建模式 → 清理旧产物 → CMake 配置 → 并行编译 → 输出验证。

以下 Mermaid 流程图展示了构建脚本的内部决策路径，帮助初学者理解每一步的输入与输出。

```mermaid
flowchart TD
    A[执行 ./build.sh] --> B{参数解析}
    B -->|release 或无参数| C[CMAKE_BUILD_TYPE=Release]
    B -->|debug| D[CMAKE_BUILD_TYPE=Debug<br/>MUSA_KERNEL_DEBUG=ON]
    C --> E[删除并重建 build/ 目录]
    D --> E
    E --> F[cmake .. 生成 Makefile]
    F --> G[make -j$(nproc) 并行编译]
    G --> H{检查 libmusa_plugin.so}
    H -->|存在| I[构建成功]
    H -->|不存在| J[构建失败退出]
```

### 构建命令

在项目根目录执行以下命令即可完成构建：

```bash
# 默认 Release 模式（推荐用于生产与常规开发）
./build.sh

# 显式指定 Release
./build.sh release

# Debug 模式（启用 Kernel 计时埋点，用于性能分析）
./build.sh debug
```

**Release 模式** 会启用 `-O3` 优化，移除所有调试开销，生成的 `libmusa_plugin.so` 具备最佳执行效率。**Debug 模式** 则开启 `MUSA_KERNEL_DEBUG` 宏，在 Kernel 内部注入计时逻辑，便于配合环境变量进行性能剖析；注意 Debug 构建仍保留 `-DNDEBUG`，以兼容 TensorFlow pip 发行版的 ABI 约定，避免因 `DCHECK` 语义不一致导致的假 refcount 崩溃。

构建成功后，终端将输出类似如下信息：

```
[SUCCESS] Build successful: libmusa_plugin.so
-rwxr-xr-x 1 user group 45M build/libmusa_plugin.so
```

Sources: [build.sh](build.sh#L1-L92), [CMakeLists.txt](CMakeLists.txt#L56-L71)

---

## 加载插件并运行验证

### 在 Python 中手动加载

构建产物 `build/libmusa_plugin.so` 是一个标准的 TensorFlow 动态插件，通过 `tf.load_library` 即可完成注册。注册成功后，TensorFlow 会自动将支持的算子分发到 MUSA 设备执行。

```python
import tensorflow as tf

# 加载 MUSA 插件
tf.load_library("./build/libmusa_plugin.so")

# 验证 MUSA 设备是否可见
print(tf.config.list_physical_devices('MUSA'))
```

若输出中包含 `PhysicalDevice(name='/physical_device:MUSA:0', device_type='MUSA')`，说明插件已成功注册且驱动层面识别到 GPU 硬件。

Sources: [README.md](README.md#L66-L69), [test/musa_test_utils.py](test/musa_test_utils.py#L36-L74)

### 运行第一个算子测试

项目内置了丰富的自动化测试，帮助初学者在修改代码前确认基线功能正常。测试框架 `test/test_runner.py` 提供了多种运行模式，下表汇总了常用参数：

| 参数 | 简写 | 说明 | 示例 |
|------|------|------|------|
| `--quiet` | `-q` | 仅显示进度条与最终摘要（默认推荐） | `python test_runner.py --quiet` |
| `--detail` | `-d` | 显示每个测试的详细执行结果 | `python test_runner.py --detail` |
| `--fusion` | `-f` | 运行融合端到端测试（`fusion/` 目录） | `python test_runner.py --fusion` |
| `--single` | - | 运行单个测试文件 | `python test_runner.py --single matmul_op_test.py` |
| `--pattern` | - | 按通配符过滤测试文件 | `python test_runner.py --pattern "*grad*_op_test.py"` |

**验证流程示例**：

```bash
cd test

# 运行所有算子测试（进度条 + 摘要）
python test_runner.py --quiet

# 单独验证 Add 算子
python test_runner.py --single ops/add_op_test.py

# 单独运行融合测试中的 LayerNorm + GeLU
python test_runner.py --single fusion/layernorm_fusion_test.py
```

测试基类 `MUSATestCase` 会在模块导入时自动搜索并加载 `../build/libmusa_plugin.so`，因此测试命令前无需手动编写 `tf.load_library`。每个算子测试通常采用 **CPU vs MUSA 结果对比** 的策略：同一组随机输入分别在 CPU 和 MUSA 设备上执行，然后使用 `assertAllClose` 校验数值一致性，容差根据 `float32`、`float16`、`bfloat16` 自动调整。

Sources: [test/test_runner.py](test/test_runner.py#L490-L578), [test/musa_test_utils.py](test/musa_test_utils.py#L84-L127), [test/ops/add_op_test.py](test/ops/add_op_test.py#L1-L98)

---

## 自动化一键验证

若希望跳过手动执行多个命令，可直接运行项目根目录下的 `test/run_all_tests.sh`。该脚本会自动检测 `build/libmusa_plugin.so` 是否存在，若不存在则先触发 `./build.sh` 进行编译，随后以安静模式运行全部算子测试并返回退出码。这对于 CI 流水线或首次环境验证尤为方便。

```bash
# 从项目根目录执行
bash test/run_all_tests.sh
```

Sources: [test/run_all_tests.sh](test/run_all_tests.sh#L1-L32)

---

## 下一步

完成快速开始并确认构建与测试通过后，建议按以下顺序深入阅读：

1. **深入理解架构** — 阅读 [项目概述与架构定位](1-xiang-mu-gai-shu-yu-jia-gou-ding-wei)，了解本扩展在 TensorFlow 生态系统中的位置，以及 Stream Executor、设备注册、Kernel 分发的整体脉络。
2. **掌握构建系统细节** — 前往 [构建系统与编译流程](4-gou-jian-xi-tong-yu-bian-yi-liu-cheng)，学习 CMake 的模块组织、`.mu` 内核编译规则、ABI 一致性处理及自定义编译选项。
3. **扩展算子或调试性能** — 若计划开发新算子，参考 [自定义 MUSA Kernel 开发指南](12-zi-ding-yi-musa-kernel-kai-fa-zhi-nan)；若需分析性能或诊断问题，查阅 [调试环境变量速查](19-diao-shi-huan-jing-bian-liang-su-cha) 与 [Kernel 计时与性能剖析](16-kernel-ji-shi-yu-xing-neng-pou-xi)。