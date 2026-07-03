# 使用说明:从下载 Docker 到查看结果

镜像已发布在 Docker Hub:**`dopafisher/energy-offloading`**
（digest `sha256:bdc1ebf31a3478f1506a263d60c2638a76a77eecf5ce9d3831de4b3eda92b876`，tag: `latest` / `v1.2` / `v1.1` / `v1.0`）

本文档适合**在你自己电脑上从零开始测试**。只需要装 Docker，不需要 Python 环境。

---

## 0. 先明确一件事：哪些要数据，哪些不要

| 想跑的东西 | 需要 `work/` 数据吗 |
|---|---|
| 推理服务 `serve` / **在线学习 `serve-online`** | **不需要**（模型已烤进镜像） |
| 训练/评估类：`task4` `task5` `online` `compare` `offload` `snakemake` `online-workflow` | **需要**（要读测量数据训练） |

> 所以：**只想体验能耗预测 + 在线学习 API → 一行命令即可，无需数据**。
> 想复现完整实验 → 需要拿到 `work/` 数据目录（约 500MB，找团队要）。

---

## 1. 安装 Docker

**Windows / macOS**：装 [Docker Desktop](https://www.docker.com/products/docker-desktop/)，装完打开，确认右下角/菜单栏 Docker 图标是绿色（运行中）。

**Linux (Ubuntu/Debian)**：
```sh
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER   # 加入 docker 组，重新登录后免 sudo
```

验证：
```sh
docker --version
# Docker version 2x.x.x ...  说明装好了
```

---

## 2. 下载镜像

```sh
docker pull dopafisher/energy-offloading:v1.2
```

下载完确认：
```sh
docker images dopafisher/energy-offloading
# 能看到 v1.2，SIZE 约 1.3GB
```

---

## 3. 最快上手：起在线学习服务（无需数据）

这是最能直接看到效果的一步 —— 一个会**在线学习**的能耗预测 API。

```sh
# 起服务（state 存到本机当前目录的 online_state/，容器重启也不丢）
mkdir -p online_state
docker run -d --name energy-online -p 8800:8800 \
  -e ONLINE_STATE_PATH=/app/models/state/online_state.pkl \
  -v "$(pwd)/online_state:/app/models/state" \
  dopafisher/energy-offloading:v1.2 serve-online
```

> Windows PowerShell 把 `$(pwd)` 换成 `${PWD}`，续行符 `\` 换成反引号 `` ` ``。

### 3.1 看服务是否正常
```sh
curl http://localhost:8800/health
# {"status": "ok"}

curl http://localhost:8800/info
# 显示 model_type=online_moe, 4 个 expert: DAW1/DAW2/Phoronix/Stress, num_updates=0
```

### 3.2 做一次预测
```sh
curl -X POST http://localhost:8800/predict \
  -H 'Content-Type: application/json' \
  -d '{"features":{"delta_cpu_ns":5e8,"delta_instructions":2e9}}'
# {"prediction_id":"...", "energy_wh":268.19, "expert":"DAW1", "model_version":0}
```
返回里：`energy_wh` = 预测能耗，`expert` = gate 自动选的专家，`prediction_id` = 这次预测的编号。

### 3.3 反馈真实能耗 → 模型在线更新
job 真正跑完后，你知道了真实能耗，回传给 `/update`：
```sh
curl -X POST http://localhost:8800/update \
  -H 'Content-Type: application/json' \
  -d '{"prediction_id":"<上一步返回的 id>","true_energy_wh":250.0}'
# {"updated_expert":"DAW1", "num_updates":1, "model_version":1}
```
`num_updates` 从 0 变 1、`model_version` 递增 → **模型真的学了这条新样本**。再 `/predict` 同样输入，结果会略有变化。

### 3.4 验证持久化（重启不丢）
```sh
docker rm -f energy-online
docker run -d --name energy-online -p 8800:8800 \
  -e ONLINE_STATE_PATH=/app/models/state/online_state.pkl \
  -v "$(pwd)/online_state:/app/models/state" \
  dopafisher/energy-offloading:v1.2 serve-online
sleep 4
curl http://localhost:8800/info    # num_updates 仍是 1，说明学到的东西保住了
```

### 3.5 停止
```sh
docker rm -f energy-online
```

---

## 4. 静态推理服务（更简单，也不需要数据）

只预测、不学习：
```sh
docker run -d --name energy-static -p 8801:8800 \
  dopafisher/energy-offloading:v1.2 serve
curl http://localhost:8801/health
curl -X POST http://localhost:8801/predict -H 'Content-Type: application/json' \
  -d '{"features":{"delta_cpu_ns":5e8,"delta_instructions":2e9}}'
docker rm -f energy-static
```

---

## 5. 跑完整实验（需要 `work/` 数据）

先把数据放到某个目录，例如 `~/work`（结构应是 `work/baseline-*/runs/*/datasets/process_interval_data.parquet`）。设一个变量指向它：

```sh
DATA=$HOME/work        # 改成你的实际路径
```

### 5.1 逐个任务

```sh
# Task 4：MoE vs 单一模型（5 折交叉验证）
docker run --rm -v $DATA:/data/work:ro dopafisher/energy-offloading:v1.2 task4 --expert linear

# Task 5：静态 vs 在线 MoE 抗漂移（Adaptive RF）
docker run --rm -v $DATA:/data/work:ro dopafisher/energy-offloading:v1.2 task5

# 7 模型统一对比表（线性/SGD/RF/LightGBM/Hoeffding/Adaptive RF...）
docker run --rm -v $DATA:/data/work:ro dopafisher/energy-offloading:v1.2 online

# 能耗感知 offloading 决策仿真
docker run --rm -v $DATA:/data/work:ro dopafisher/energy-offloading:v1.2 offload --expert linear

# 真实 Snakemake 调度 DAG（5 策略对比）
docker run --rm -v $DATA:/data/work:ro dopafisher/energy-offloading:v1.2 snakemake compare
```

结果直接打印在终端（R²、MAE、对比表）。

### 5.2 把结果存到本机（挂 results 卷）

```sh
mkdir -p results
docker run --rm \
  -v $DATA:/data/work:ro \
  -v "$(pwd)/results:/app/results" \
  dopafisher/energy-offloading:v1.2 online
# 跑完在本机 ./results/ 下能看到 CSV / 图 / md
ls results/moe/
```

### 5.3 在线学习闭环（自动 predict + update，无需手动 curl）

```sh
mkdir -p online_state results
docker run --rm \
  -v $DATA:/data/work:ro \
  -v "$(pwd)/online_state:/app/models/state" \
  -v "$(pwd)/results:/app/results" \
  dopafisher/energy-offloading:v1.2 online-workflow --jobs-per-workload 5 \
  --out /app/results/online_workflow_run.json

# 看闭环结果
cat results/online_workflow_run.json
```
这个会逐个 job：预测 → “执行” → 用真实能耗更新模型，最后给出汇总（num_updates 增长、误差前后半段下降）。

### 5.4 新架构 resource-MoE（按 CPU/Memory/IO/Network 分专家）

```sh
docker run --rm --entrypoint bash -v $DATA:/data/work:ro \
  dopafisher/energy-offloading:v1.2 \
  -lc "PYTHONPATH=. python -m feature_moe.run_resource_moe --expert rf"
```

---

## 6. 怎么看结果

- **终端输出**：task4/task5/online/offload 跑完直接打印 R²、MAE、对比表、gate 权重。
- **本机 `results/` 目录**（挂了 5.2 的卷之后）：
  - `results/moe/online_baseline_comparison.md` — 7 模型统一对比表
  - `results/moe/ONLINE_BASELINES_REPORT.md` — 在线 baseline 完整报告
  - `results/moe/*.png` — 各种对比图
  - `results/feature_moe/cmp_rf.png` — resource-MoE 对比图
  - `results/online_workflow_run.json` — 闭环逐 job 记录
- **在线服务**：`/info` 看 `num_updates` / `model_version` 是否增长，判断在线学习是否在生效。

---

## 7. 一键自检（可选，确认一切正常）

如果你有完整项目源码（不只是镜像），可以跑三个 smoke test：
```sh
./deploy/smoke_test.sh                  # 静态推理
./deploy/online_smoke_test.sh           # 在线 API + 重启持久化
./deploy/online_workflow_smoke_test.sh  # 闭环
```
只有镜像的话，第 3~5 节的 curl / docker run 就是等价的手动自检。

---

## 8. 项目满足哪些要求 + 诚实边界

**已实现并验证**：
- 进程级能耗预测（线性 baseline 复现 + 7 模型扩展对比）
- Workload-grouped MoE + 在线学习（Task 4/5，Online-MoE 抗漂移 R² 0.93）
- Feature-based（资源分组）MoE 新架构
- 能耗感知 offloading 决策（MoE 节能 6.2%）
- 真实 Snakemake 调度集成
- **Live online learning 服务 + 自动闭环**（predict→update→持久化→重启恢复，全部实测）

**诚实边界（不夸大）**：
- Offloading / 多集群是**仿真**，无真实第二物理集群
- 闭环里的“真实能耗”来自**实测数据模拟** job 完成信号，非运行时实时测量
- K8s manifest（`deploy/k8s/`）是**可用模板，未真正 apply**（当时无集群）
- resource-MoE 在 CPU 主导数据上收益有限，workload-MoE 仍最强

---

## 9. 常见问题

- **端口被占用**：把 `-p 8800:8800` 改成别的，如 `-p 9000:8800`，然后 curl 用 `localhost:9000`。
- **`docker: permission denied`**（Linux）：`sudo` 前缀，或把用户加入 docker 组后重登录。
- **训练任务报 `No workload parquet found`**：说明没挂 `work/` 数据，或挂载路径不对（必须是 `-v <你的work路径>:/data/work:ro`）。
- **容器还在后台跑**：`docker ps` 看，`docker rm -f <名字>` 停掉。
