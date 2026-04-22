import os

out_path = "/home/aero/桌面/知识学习/GPU计算生态完全指南.md"

sections = []

# Section 10: MUSA Toolkit and first program
sections.append("""### 3.5 MUSA Toolkit——国产开发工具箱

#### 3.5.1 Toolkit组件

MUSA Toolkit与CUDA Toolkit组件一一对应：

| 组件 | CUDA名称 | MUSA名称 | 作用 |
|------|----------|----------|------|
| 编译器 | nvcc | mcc | 编译.cu/.mu文件 |
| 反汇编 | cuobjdump | muobjdump | 查看二进制文件 |
| 性能分析 | nvprof/nsys | musa-prof | 性能分析 |
| 调试器 | cuda-gdb | musa-gdb | GPU调试 |
| 数学库 | cuBLAS | muBLAS | 线性代数 |
| 深度学习库 | cuDNN | muDNN | 深度学习加速 |
| 随机数 | cuRAND | muRAND | 随机数生成 |
| FFT | cuFFT | muFFT | 快速傅里叶变换 |

#### 3.5.2 mcc编译器

mcc是MUSA的**核心编译器**，使用方法与nvcc几乎相同：

```bash
# 基础编译
mcc -o program program.mu

# 指定架构
mcc -arch=mp_21 -o program program.mu

# 调试版本
mcc -g -G -o program_debug program.mu
```

**mcc常用选项**：

| 选项 | 作用 | 与nvcc对比 |
|------|------|-----------|
| `-o <文件名>` | 指定输出文件名 | 相同 |
| `-arch=mp_21` | 指定MUSA架构 | 类似 `-arch=sm_80` |
| `-O3` | 最高优化级别 | 相同 |
| `-g` | 包含调试信息 | 相同 |
| `-G` | 设备代码调试 | 相同 |

#### 3.5.3 第一个完整的MUSA程序

```cpp
#include <musa_runtime.h>
#include <stdio.h>

// 核函数：在MUSA GPU上执行
// __global__ 修饰符与CUDA完全相同
__global__ void vector_add(const float *a, const float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

void check_musa(musaError_t err, const char *file, int line) {
    if (err != musaSuccess) {
        printf("MUSA错误 在 %s:%d - %s\n", file, line, musaGetErrorString(err));
        exit(1);
    }
}
#define MUSA_CHECK(call) check_musa(call, __FILE__, __LINE__)

int main() {
    int n = 1000000;
    size_t size = n * sizeof(float);
    
    printf("===== 向量加法 MUSA 程序 =====\n");
    printf("数据规模: %d 个浮点数\n\n", n);
    
    // 1. 主机内存分配
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);
    
    // 2. 初始化数据
    for (int i = 0; i < n; i++) {
        h_a[i] = (float)i;
        h_b[i] = (float)(n - i);
    }
    printf("✓ 主机数据初始化完成\n");
    
    // 3. 设备内存分配
    float *d_a, *d_b, *d_c;
    MUSA_CHECK(musaMalloc((void**)&d_a, size));
    MUSA_CHECK(musaMalloc((void**)&d_b, size));
    MUSA_CHECK(musaMalloc((void**)&d_c, size));
    printf("✓ 设备内存分配完成\n");
    
    // 4. 数据传输
    MUSA_CHECK(musaMemcpy(d_a, h_a, size, musaMemcpyHostToDevice));
    MUSA_CHECK(musaMemcpy(d_b, h_b, size, musaMemcpyHostToDevice));
    printf("✓ 数据复制到设备完成\n");
    
    // 5. 启动核函数
    int threads_per_block = 256;
    int blocks = (n + threads_per_block - 1) / threads_per_block;
    
    printf("\n启动配置:\n");
    printf("  块数量: %d\n", blocks);
    printf("  每块线程数: %d\n", threads_per_block);
    printf("  总线程数: %d\n\n", blocks * threads_per_block);
    
    vector_add<<<blocks, threads_per_block>>>(d_a, d_b, d_c, n);
    MUSA_CHECK(musaGetLastError());
    printf("✓ 核函数已启动\n");
    
    // 6. 传回结果
    MUSA_CHECK(musaMemcpy(h_c, d_c, size, musaMemcpyDeviceToHost));
    printf("✓ 结果复制回主机完成\n");
    
    // 7. 验证
    int correct = 1;
    for (int i = 0; i < n; i++) {
        float expected = h_a[i] + h_b[i];
        if (h_c[i] != expected) {
            correct = 0;
            printf("验证失败 在索引 %d\n", i);
            break;
        }
    }
    printf("✓ 结果验证%s\n", correct ? "通过" : "失败");
    
    // 8. 释放资源
    free(h_a); free(h_b); free(h_c);
    MUSA_CHECK(musaFree(d_a));
    MUSA_CHECK(musaFree(d_b));
    MUSA_CHECK(musaFree(d_c));
    printf("✓ 所有资源已释放\n");
    
    return 0;
}
```

**编译运行**：
```bash
mcc -o vector_add_musa vector_add_musa.mu
./vector_add_musa
```

**CUDA vs MUSA代码对比**：

| 元素 | CUDA代码 | MUSA代码 |
|------|----------|----------|
| 头文件 | `#include <cuda_runtime.h>` | `#include <musa_runtime.h>` |
| 错误类型 | `cudaError_t` | `musaError_t` |
| 成功标志 | `cudaSuccess` | `musaSuccess` |
| 内存分配 | `cudaMalloc()` | `musaMalloc()` |
| 内存复制 | `cudaMemcpy()` | `musaMemcpy()` |
| 同步 | `cudaDeviceSynchronize()` | `musaDeviceSynchronize()` |
| 错误检查 | `cudaGetLastError()` | `musaGetLastError()` |
| 核函数语法 | `<<<blocks, threads>>>` | `<<<blocks, threads>>>` |

**结论**：只需简单的"查找替换"，CUDA代码即可在MUSA上运行！

""")

with open(out_path, "a", encoding="utf-8") as f:
    for section in sections:
        f.write(section)

print("Section 10 appended")
