本文面向具备基础 C/C++ 知识的初级开发者，目标是让你在阅读后的 30 分钟内，理解编写和运行第一个 GPU 程序所需的全部核心要素。我们不会在此深入硬件架构或库函数优化，而是聚焦于**一个可运行的程序、一套固定的工作流程、以及一份清晰的环境准备清单**。理解了这三样东西，你就已经跨过了 GPU 编程的最低门槛，可以带着明确的目标进入后续章节的深度学习。

Sources: [GPU计算生态完全指南.md](GPU计算生态完全指南.md#L1-L8)

## GPU 程序的"Hello World"：向量加法

在 CPU 编程中，"Hello World"是一行打印语句；在 GPU 编程中，**向量加法**承担着同样的角色。它的逻辑足够简单——两个数组对应元素相加——但完整展示了 GPU 程序区别于普通 C++ 程序的所有关键环节：Kernel 函数声明、主机与设备内存分离、数据拷贝、以及 `<<<...>>>` 执行配置语法。以下是一个完整的 CUDA 向量加法程序，你可以直接复制、编译并运行。

```cpp
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void vector_add(float* out, const float* a, const float* b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}

int main() {
    const int n = 1024;
    const int size = n * sizeof(float);

    // 1. 主机内存分配与初始化
    float* h_a = (float*)malloc(size);
    float* h_b = (float*)malloc(size);
    float* h_out = (float*)malloc(size);
    for (int i = 0; i < n; i++) {
        h_a[i] = (float)i;
        h_b[i] = (float)(n - i);
    }

    // 2. 设备内存分配
    float* d_a; cudaMalloc(&d_a, size);
    float* d_b; cudaMalloc(&d_b, size);
    float* d_out; cudaMalloc(&d_out, size);

    // 3. 数据从主机拷贝到设备
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // 4. 启动 Kernel
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    vector_add<<<blocks, threads>>>(d_out, d_a, d_b, n);

    // 5. 结果从设备拷贝回主机
    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);
    printf("Result: %f + %f = %f\n", h_a[0], h_b[0], h_out[0]);

    // 6. 释放资源
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_out);
    free(h_a); free(h_b); free(h_out);

    return 0;
}
```

**编译命令**：
```bash
nvcc -o vector_add vector_add.cu
./vector_add
```

Sources: [GPU计算生态完全指南.md](GPU计算生态完全指南.md#L1728-L1785)

## 所有 GPU 程序都遵循的六步工作流程

无论你的程序是简单的向量加法，还是复杂的深度学习推理，所有 GPU 程序都遵循同一套**"主机准备 → 设备执行 → 主机收尾"**的固定模式。理解这个模式，你就掌握了阅读任何 GPU 代码的骨架钥匙。

```mermaid
flowchart LR
    A[① 分配并初始化<br/>主机内存] --> B[② 分配<br/>设备内存]
    B --> C[③ 主机 → 设备<br/>数据传输]
    C --> D[④ 启动 Kernel<br/><<<blocks, threads>>>]
    D --> E[⑤ 设备 → 主机<br/>结果回传]
    E --> F[⑥ 释放主机与<br/>设备资源]
    
    style D fill:#e8f5e9
```

| 步骤 | 动作 | 对应 API 示例 | 关键认知 |
|------|------|--------------|---------|
| ① 主机准备 | 在 CPU 内存中分配数组并填充数据 | `malloc()` | 这是普通的 C/C++ 操作，GPU 尚不参与 |
| ② 设备分配 | 在 GPU 显存中申请同样大小的空间 | `cudaMalloc()` | GPU 有自己的独立内存，不能直接访问 CPU 内存中的变量 |
| ③ 数据上载 | 将数据从 CPU 复制到 GPU | `cudaMemcpy(..., cudaMemcpyHostToDevice)` | 这是性能敏感点，PCIe 传输有开销 |
| ④ Kernel 启动 | 配置线程网格并启动 GPU 函数 | `<<<blocks, threads>>>` | `__global__` 修饰的函数在 GPU 上执行，由成千上万线程并行运行 |
| ⑤ 结果下载 | 将计算结果从 GPU 复制回 CPU | `cudaMemcpy(..., cudaMemcpyDeviceToHost)` | 不执行这一步，主机无法读取 GPU 的计算结果 |
| ⑥ 资源释放 | 分别释放 GPU 和 CPU 内存 | `cudaFree()` + `free()` | GPU 内存不会随程序退出自动完全回收，必须显式释放 |

这六个步骤构成了 GPU 编程的**最小完整闭环**。任何省略了其中一步的代码都存在隐患：跳过步骤②会导致向非法地址写入，跳过步骤⑤会得到未定义的结果，跳过步骤⑥则会造成显存泄漏。

Sources: [GPU计算生态完全指南.md](GPU计算生态完全指南.md#L1745-L1785)

## 从 CUDA 到 MUSA：几乎相同的体验

如果你使用的是国产摩尔线程 GPU，上面的六步工作流程完全不变，你只需要修改**前缀和编译器**。MUSA 的设计目标就是与 CUDA 保持高度兼容，让已有的 CUDA 代码能够平滑迁移。以下是将上面的 CUDA 程序改为 MUSA 程序所需的全部改动：

| 改动项 | CUDA 写法 | MUSA 写法 |
|--------|----------|----------|
| 头文件 | `#include <cuda_runtime.h>` | `#include <musa_runtime.h>` |
| 内存分配 | `cudaMalloc(&ptr, size)` | `musaMalloc(&ptr, size)` |
| 内存拷贝 | `cudaMemcpy(...)` | `musaMemcpy(...)` |
| 内存释放 | `cudaFree(ptr)` | `musaFree(ptr)` |
| 同步函数 | `cudaDeviceSynchronize()` | `musaDeviceSynchronize()` |
| 错误检查 | `cudaGetLastError()` | `musaGetLastError()` |
| 编译器 | `nvcc -o program program.cu` | `mcc -o program program.cu` |
| Kernel 语法 | `<<<blocks, threads>>>` | `<<<blocks, threads>>>`（完全相同） |
| 修饰符 | `__global__` | `__global__`（完全相同） |

**核心结论**：CUDA 和 MUSA 在编程模型、内存管理语义和 Kernel 启动方式上是**同构的**。学习 CUDA 就是在同时学习 MUSA，区别仅限于 API 前缀和编译器命令。如果你希望看到更详细的代码对比，可以参考[基础向量加法：CUDA与MUSA对比](21-ji-chu-xiang-liang-jia-fa-cudayu-musadui-bi)。

Sources: [GPU计算生态完全指南.md](GPU计算生态完全指南.md#L1846-L1857)
Sources: [GPU计算生态完全指南.md](GPU计算生态完全指南.md#L67-L81)

## 环境准备清单

在运行上面的程序之前，你需要安装对应的 Toolkit。以下是不同场景的最小安装要求：

| 目标平台 | 必须安装 | 可选安装 | 说明 |
|----------|---------|---------|------|
| NVIDIA CUDA | CUDA Toolkit（含 nvcc + Runtime + Driver） | cuDNN、cuBLAS、NCCL | Toolkit 本身已包含编译器、Runtime 和基础数学库，足以编译运行向量加法 |
| 摩尔线程 MUSA | MUSA Toolkit（含 mcc + Runtime + Driver） | muDNN、muBLAS、MCCL | 组件与 CUDA Toolkit 一一对应，安装后即可编译 MUSA 程序 |

**给初学者的建议**：如果你只是想理解 GPU 编程的基本原理并运行示例代码，**只需要安装 Toolkit 即可**。cuDNN 和 muDNN 等深度学习库可以等到你进入深度学习相关开发时再安装。详细的版本匹配策略和安装步骤，请参考[版本匹配与安装策略](20-ban-ben-pi-pei-yu-an-zhuang-ce-lue)。

Sources: [GPU计算生态完全指南.md](GPU计算生态完全指南.md#L422-L450)
Sources: [GPU计算生态完全指南.md](GPU计算生态完全指南.md#L2018-L2020)

## 继续探索

完成本章后，你已经知道了 GPU 程序长什么样、遵循什么流程、以及 CUDA 与 MUSA 的关系。接下来建议按以下顺序深入：

1. **[GPU计算生态全景图](3-gpuji-suan-sheng-tai-quan-jing-tu)** — 将本文的六步程序放入整个生态的五层架构中，理解你的代码究竟调用了哪些层级
2. **[餐厅类比：理解GPU生态层次](4-can-ting-lei-bi-li-jie-gpusheng-tai-ceng-ci)** — 用更生动的类比加深对层级依赖的直觉记忆
3. **[GPU与CPU的核心差异](5-gpuyu-cpude-he-xin-chai-yi)** — 从硬件设计哲学出发，理解为什么 GPU 需要 `<<<...>>>` 这样的执行模型
4. **[CUDA与MUSA：两大生态概览](6-cudayu-musa-liang-da-sheng-tai-gai-lan)** — 如果你同时使用或对比两个生态，这是进入 Deep Dive 前的最后一站概览

当你建立了清晰的全局认知后，可以进入 **Deep Dive** 部分，选择你感兴趣的层级进行深入研究：硬件架构、驱动与运行时、内存管理、编译器、数学库或深度学习库。每一层都有完整的代码示例等待你编译和修改。