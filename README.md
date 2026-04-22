# TensorFlow MUSA Extension Wiki

本项目是 TensorFlow MUSA Extension 的完整文档集合，包含 22 篇技术文档，涵盖架构设计、算子实现、图优化、调试与测试等全链路技术细节。

## 📁 文件说明

| 文件/目录 | 说明 |
|-----------|------|
| `2026-04-22-120026/` | 原始 Markdown 文档目录（22 篇） |
| `combined.md` | 合并后的完整 Markdown 文档 |
| `wiki.html` | 带侧边栏导航的 HTML 版本（推荐浏览） |
| `generate_wiki.py` | 生成 HTML 的 Python 脚本 |

## 🚀 快速使用

### 方式一：直接打开 HTML（推荐）
```bash
# Linux
xdg-open wiki.html

# macOS
open wiki.html

# Windows
start wiki.html
```

### 方式二：查看合并文档
```bash
# 用任意 Markdown 编辑器打开
combined.md
```

### 方式三：查看原始文档
```bash
# 原始文档在 2026-04-22-120026/ 目录下
ls 2026-04-22-120026/
```

## 📚 文档结构

- **Get Started**（入门）
  - 项目概述与架构定位
  - 快速开始
  - 环境依赖与前置准备
  - 构建系统与编译流程

- **Deep Dive**（深入）
  - 设备运行时架构（Stream Executor、内存管理、Kernel 注册）
  - 算子实现体系（数学、神经网络、训练优化器、自定义 Kernel）
  - 图优化与融合（Grappler、融合模式、自动混合精度）
  - 调试与性能分析（计时、遥测、内存诊断、环境变量）
  - 测试与质量保障（测试框架、算子测试、端到端测试）

## 🛠️ 重新生成 HTML

```bash
python3 generate_wiki.py
```

## 📄 生成信息

- **生成时间**: 2026-04-22
- **文档总数**: 22 篇
- **语言**: 中文
