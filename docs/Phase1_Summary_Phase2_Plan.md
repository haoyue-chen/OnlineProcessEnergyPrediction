# 分布式系统项目阶段性总结与后续计划（中文版）

## 1. 项目背景

本项目来源于 TU Berlin 分布式系统研究组的能源感知工作流调度研究。

整体研究路线如下：

### 第一阶段：Process-Level Energy Prediction

根据节点级功耗数据（Node-Level Power Consumption）以及进程级监控指标（Process Metrics），建立能源预测模型，实现：

```text
Process Metrics
        ↓
Energy Prediction Model
        ↓
Predicted Process Energy
```

该部分对应论文：

* Learning Process Energy from Node-Level Data

---

### 第二阶段：Online Learning / Dynamic Modeling

研究模型在运行过程中持续更新（Online Learning）的可行性，并分析：

* 哪些工作负载最受益于在线更新
* 为什么不同工作负载表现不同
* 是否需要多个专家模型（Mixture of Experts）

---

### 第三阶段：Middleware Integration

将最终模型集成到 Snakemake Offloading Middleware 中，用于跨集群资源调度：

```text
Workflow
    ↓
Energy Prediction
    ↓
Offloading Decision
    ↓
Cluster Selection
```

目标：

* 降低能耗
* 降低执行成本
* 提升 Makespan

---

# 2. 已完成工作

---

## 2.1 系统理解与环境复现

已完成：

### 阅读与理解

* Adaptive Offloading Middleware
* Learning Process Energy from Node-Level Data
* 项目代码仓库

理解了当前系统的完整工作流程：

```text
Monitoring
    ↓
Feature Extraction
    ↓
Process-Power Mapping
    ↓
Linear Regression
    ↓
Energy Prediction
```

目前系统属于：

```text
Offline Training
+
Static Linear Model
```

即：

* 数据采集
* 模型训练
* 模型评估

三个阶段相互独立。

---

## 2.2 Workload 数据采集

完成了数据采集工作。

通过：

* Monitoring Script
* Power Meter

在服务器上运行多个 Benchmark Workload。

目前获得的数据集：

* DAW1
* DAW2
* Phoronix
* Stress

采集内容：

### Process Metrics

例如：

* CPU Usage
* Memory Usage
* IO Statistics
* Context Switches
* Threads
* Process Runtime

等特征。

### Node-Level Power

通过 Power Meter 获取节点级功耗。

最终得到：

```text
Process Metrics
      ↔
Interval Power
```

对应关系数据集。

---

## 2.3 特征分析

完成：

### Correlation Analysis

分析各 Process Metrics 与 Power 的相关性。

目的：

识别最有影响力的特征。

---

### Sequential Feature Selection (SFS)

进一步寻找：

```text
Best Feature Combination
```

减少冗余特征。

提高模型可解释性与泛化能力。

---

## 2.4 Baseline 模型复现

使用原论文方法：

### CVXPY Estimator

完成：

```text
Process Metrics
       ↓
Linear Mapping
       ↓
Power Prediction
```

复现论文中的线性回归结果。

---

# 3. Baseline 扩展实验

为了验证是否存在更优预测模型，我们进一步测试了多种机器学习模型。

测试模型：

* Linear Regression
* SGD Regressor
* Random Forest
* LightGBM

评估指标：

R² Score

---

## 实验结果

| Dataset  | Linear | SGD   | RF    | LightGBM |
| -------- | ------ | ----- | ----- | -------- |
| DAW1     | 0.707  | 0.706 | 0.932 | 0.931    |
| DAW2     | 0.846  | 0.846 | 0.964 | 0.965    |
| Phoronix | 0.730  | 0.647 | 0.874 | 0.862    |
| Stress   | 0.864  | 0.864 | 0.963 | 0.966    |

---

# 4. 当前发现（Current Findings）

目前的观察不仅是“哪个模型更好”，更重要的是解释背后的原因。

---

## Finding 1

### SGD ≈ Linear Regression

在所有数据集上：

```text
SGD ≈ Linear
```

表现基本一致。

原因：

SGD 本质上仍然是线性模型。

只是：

```text
Batch Optimization
↓
变为
Incremental Optimization
```

模型表达能力没有提升。

说明：

当前系统的瓶颈并不在优化方法，而在模型结构本身。

---

## Finding 2

### Tree-Based Models 显著优于 Linear Models

Random Forest 和 LightGBM 在所有数据集上均明显优于线性模型。

提升幅度：

约：

```text
+10% ~ +25%
```

R² Improvement。

说明：

Process Metrics 与 Energy Consumption 之间存在明显的非线性关系。

当前论文中的线性映射：

```text
Power
=
a₁x₁+a₂x₂+...
```

无法充分描述真实能耗行为。

---

## Finding 3

### 不同 Workload 的可预测性不同

DAW 数据集效果最好：

```text
R² > 0.93
```

而：

Phoronix 明显较低。

说明：

不同工作负载具有不同的资源行为模式。

可能包括：

* CPU Bound
* Memory Bound
* IO Bound

等不同特征。

这为后续 Mixture-of-Experts 提供了动机。

---

## Finding 4

### Random Forest ≈ LightGBM

两者结果非常接近。

说明：

当前数据规模下：

模型复杂度已经不是主要限制因素。

未来改进方向可能不是：

```text
更复杂模型
```

而是：

```text
动态模型选择
+
在线更新机制
```

---

# 5. 当前问题

目前系统仍然存在以下限制：

---

## 问题1

静态训练

模型训练完成后：

```text
Model Fixed
```

无法适应：

* Workload Drift
* Resource Changes
* Runtime Variations

---

## 问题2

单一模型

所有 Workload 使用同一个模型。

假设：

```text
One Model Fits All
```

但实验结果表明：

不同 Workload 可能具有不同能耗模式。

---

## 问题3

Pipeline 分离

目前：

```text
Monitoring
Training
Evaluation
```

完全分离。

不符合 Online Learning 的要求。

---

# 6. 下一阶段目标（Objective 2）

重点研究：

## Online Learning

以及

## Mixture of Experts (MoE)

---

# 7. 文献调研任务

---

## Paper Group A：Online Learning

重点阅读：

### River

Online Machine Learning Library

重点关注：

* SGD Regressor
* Adaptive Regression
* Hoeffding Tree
* Adaptive Random Forest

关注问题：

1. 如何持续更新模型？
2. 如何处理概念漂移（Concept Drift）？
3. 如何评估在线模型？

---

## Paper Group B：MoE

重点阅读：

### Sizey

研究内容：

动态选择不同预测模型。

重点关注：

1. Expert 如何定义？
2. Gating 如何设计？
3. Workload 如何分类？
4. 如何动态路由数据？

---

# 8. 计划中的模型架构

当前设想：

```text
Process Metrics
        ↓
Gate
        ↓
Expert Selection
        ↓
Expert Model
        ↓
Energy Prediction
```

可能形式：

```text
DAW Expert

Phoronix Expert

Stress Expert
```

或者：

```text
CPU Expert

Memory Expert

IO Expert
```

需要进一步实验验证。

---

# 9. MoE + Online Learning 方案

计划设计：

```text
Monitoring
      ↓
Feature Extraction
      ↓
Gate
      ↓
Select Expert
      ↓
Prediction
      ↓
Receive New Sample
      ↓
Online Update
```

即：

先选择最合适 Expert。

再对当前 Expert 进行 Online Learning。

该方案也是目前与其他组相比可能具有创新性的方向。

---

# 10. Objective 3 规划

最终目标：

将模型接入：

Snakemake Offloading Middleware

形成：

```text
Workflow
      ↓
Energy Prediction
      ↓
Decision Engine
      ↓
Cluster Selection
```

例如：

```text
Predicted Energy > Threshold
      ↓
Offload to Cluster B
```

或者：

综合考虑：

* Runtime
* Energy
* Cost

进行调度决策。

---

# 11. 未来两周任务安排

## Task 1

重新阅读 Sizey。

输出：

* Expert Design
* Gating Strategy
* 可迁移思想总结

负责人：

待分配

---

## Task 2

调研 Online Learning 方法。

重点：

* River
* Hoeffding Tree
* Adaptive Random Forest

负责人：

待分配

---

## Task 3

设计 MoE 架构。

输出：

系统设计图。

负责人：

待分配

---

## Task 4

实现 MoE Baseline。

目标：

验证：

```text
MoE
vs
Single Model
```

负责人：

待分配

---

## Task 5

实现 Online Update。

目标：

验证：

```text
MoE
vs
MoE + Online Learning
```

负责人：

待分配

---

# 当前项目进度评估

已完成：

Objective 1 ≈ 70%

整体项目：

≈ 40%

下一阶段核心工作：

MoE + Online Learning 设计与验证。
