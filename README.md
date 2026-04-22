# knowledge-learning

本仓库包含多个知识模块，每个模块都使用 [ZRead](https://github.com/ZreadAI/zread) 工具进行管理和阅读。

## 仓库结构

| 目录 | 说明 | ZRead 打开方式 |
|------|------|----------------|
| [`内存管理/`](内存管理/) | GPU 内存管理深度教程 | `cd 内存管理 && zread browse` |
| [`GPU生态指南/`](GPU生态指南/) | GPU 计算生态（CUDA / MUSA）全景指南 | `cd GPU生态指南 && zread browse` |
| [`horovod/`](horovod/) | Horovod + MCCL 分布式训练指南 | `cd horovod && zread browse` |
| [`wiki-tf_musa_ext/`](wiki-tf_musa_ext/) | TensorFlow MUSA 扩展（tf_musa_ext）Wiki | `cd wiki-tf_musa_ext && zread browse` |

## 使用 ZRead 阅读

### 1. 安装 ZRead CLI

你可以通过 npm、homebrew 或 winget 安装：

**npm 安装**（适用于 Windows、Linux 和 macOS）
```bash
npm install -g zread_cli
```

**homebrew 安装**（适用于 Linux 和 macOS）
```bash
brew tap ZreadAI/homebrew-tap
brew install zread
```

**winget 安装**（适用于 Windows 10/11）
```bash
winget install ZhipuAI.Zread
```

### 2. 阅读各模块

进入任意模块目录后，直接运行 `zread`，CLI 会根据当前状态推荐下一步操作：

```bash
# 阅读 GPU 内存管理教程
cd 内存管理
zread

# 阅读 GPU 生态指南
cd GPU生态指南
zread

# 阅读 Horovod 分布式训练指南
cd horovod
zread

# 阅读 TensorFlow MUSA 扩展 Wiki
cd wiki-tf_musa_ext
zread
```

如果已经生成过文档，想在浏览器中直接打开：

```bash
cd <目录名>
zread browse
```

### 3. ZRead 工作原理

ZRead 会查找目录下的 `.zread/wiki/` 路径：

- **草稿模式**：读取 `.zread/wiki/drafts/wiki.json` 获取文章列表
- **版本模式**：读取 `.zread/wiki/current` 获取当前版本号，然后加载 `.zread/wiki/versions/<version>/` 下的内容

例如：
- `内存管理/.zread/wiki/current` 指向 `versions/2026-04-22-155935`
- `GPU生态指南/.zread/wiki/drafts/wiki.json` 直接定义草稿页面列表

生成完成后，文档会保存在项目目录下的 `.zread/` 中：

```
.zread/
  state.json
  wiki/
    current              # 当前版本文档
    versions/            # 历史版本文档
    drafts/              # 生成完成前的草稿文档
```

### 4. 常用 ZRead 命令

| 命令 | 功能说明 |
|------|----------|
| `zread` | CLI 默认入口，根据当前环境自动推荐下一步操作 |
| `zread generate` | 为当前目录生成项目文档 |
| `zread browse` | 在浏览器打开当前项目已生成的文档 |
| `zread login` | 登录账号或配置 API key |
| `zread config` | 查看或修改 CLI 配置 |
| `zread update` | 更新 CLI 到最新版本 |
| `zread version` | 查看当前 CLI 版本 |

如需查看完整帮助信息，可以运行：

```bash
zread --help
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

各模块的内容通过 ZRead 工具生成和管理。如需更新，请进入对应目录后使用 ZRead 的生成命令重新构建 wiki 内容：

```bash
cd <目录名>
zread generate
```

---

*本仓库使用 ZRead 进行结构化知识管理。*
