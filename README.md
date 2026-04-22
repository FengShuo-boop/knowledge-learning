# knowledge-learning

本仓库包含多个知识模块，每个模块都使用 [ZRead](https://github.com/FengShuo-boop/zread) 工具进行管理和阅读。

## 仓库结构

| 目录 | 说明 | ZRead 打开方式 |
|------|------|----------------|
| [`内存管理/`](内存管理/) | GPU 内存管理深度教程 | `zread 内存管理` |
| [`GPU生态指南/`](GPU生态指南/) | GPU 计算生态（CUDA / MUSA）全景指南 | `zread GPU生态指南` |
| [`horovod/`](horovod/) | Horovod + MCCL 分布式训练指南 | `zread horovod` |
| [`wiki-tf_musa_ext/`](wiki-tf_musa_ext/) | TensorFlow MUSA 扩展（tf_musa_ext）Wiki | `zread wiki-tf_musa_ext` |

## 使用 ZRead 阅读

### 1. 安装 ZRead

确保已安装 ZRead 工具：

```bash
# 从源码安装
git clone https://github.com/FengShuo-boop/zread.git
cd zread
pip install -e .
```

### 2. 阅读各模块

在仓库根目录下，直接使用目录名打开对应的知识模块：

```bash
# 阅读 GPU 内存管理教程
zread 内存管理

# 阅读 GPU 生态指南
zread GPU生态指南

# 阅读 Horovod 分布式训练指南
zread horovod

# 阅读 TensorFlow MUSA 扩展 Wiki
zread wiki-tf_musa_ext
```

### 3. ZRead 工作原理

ZRead 会查找目录下的 `.zread/wiki/` 路径：

- **草稿模式**：读取 `.zread/wiki/drafts/wiki.json` 获取文章列表
- **版本模式**：读取 `.zread/wiki/current` 获取当前版本号，然后加载 `.zread/wiki/versions/<version>/` 下的内容

例如：
- `内存管理/.zread/wiki/current` 指向 `versions/2026-04-22-155935`
- `GPU生态指南/.zread/wiki/drafts/wiki.json` 直接定义草稿页面列表

### 4. 常用 ZRead 命令

```bash
# 交互式浏览（默认）
zread <目录名>

# 查看帮助
zread --help

# 指定特定版本（如果支持）
zread <目录名> --version <版本号>
```

## 各模块简介

### 内存管理
GPU 内存管理深度教程，涵盖：
- GPU 硬件内存层次解析
- CUDA 内存 API 全景与选型
- 内存分配全链路（从 `cudaMalloc` 到驱动）
- 训练/推理场景内存优化策略
- 多 GPU、多进程与多租户环境

### GPU生态指南
GPU 计算生态全景指南，涵盖：
- GPU 与 CPU 的核心差异
- CUDA 生态详解（硬件架构、驱动、Toolkit、cuDNN、cuBLAS、NCCL）
- MUSA 生态详解（架构设计、兼容性、muDNN、muBLAS、MCCL）
- CUDA 到 MUSA 的迁移策略与代码对比

### horovod
Horovod 分布式训练与 MCCL 通信库指南：
- Horovod 安装与配置
- MCCL（MUSA Collective Communications Library）使用说明
- 分布式训练最佳实践

### wiki-tf_musa_ext
TensorFlow MUSA 扩展（tf_musa_ext）Wiki：
- 项目概述与架构定位
- Stream Executor 与设备注册机制
- Kernel 注册与算子分发流程
- 算子融合、图优化与性能剖析
- 测试框架与调试工具

## 贡献与更新

各模块的内容通过 ZRead 工具生成和管理。如需更新，请使用 ZRead 的生成命令重新构建 wiki 内容。

---

*本仓库使用 ZRead 进行结构化知识管理。*
