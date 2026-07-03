# Energy-Offloading — Feature/Workload MoE 能耗预测与在线学习系统

基于进程级资源特征(Process Metrics)预测科学工作流的节点能耗,并用预测结果做能耗感知的跨集群调度决策,同时支持在线学习闭环。来自 TU Berlin 分布式系统研究组的能源感知工作流调度研究。

> **数据来源**:测量数据在 `../work/`(4 个 workload:DAW1 / DAW2 / Phoronix / Stress),由上游采集项目 `ProcessEnergyAccounting/` 产生,本项目**只读取**。运行时挂载到 `/data/work`。

---

## 实现了哪些功能

### 1. 进程级能耗预测(Phase 1,复现 + 扩展)
- 按 interval 聚合进程特征(求和)→ X,`interval_energy` → y
- 复现论文 CVXPY 线性 baseline + 扩展到多模型对比(linear / rf / extra_trees / hgb(LightGBM 替身)/ knn / mlp / svr)
- 与中期报告(`TeamGreen-MidTerm-Presentation.pdf`)的 baseline 在 5 折 CV 下复现一致

### 2. Workload-grouped MoE + Online Learning(Phase 2,Task 3/4/5)
- **MoE**:gate(RandomForest 分类器)把每个 interval 路由到对应 workload 的 expert
- **Task 4**(MoE vs 单一模型):5 折 CV,linear 专家下 MoE 提升 +0.10~0.17 R²
- **Task 5**(在线学习抗漂移):工作负载串流制造 drift,Online-MoE(ARF)R² 0.93 / MAE 6.2,远超冻结模型
- **6 个 online 模型对比**:linear(SGD+Adam)/ pa / Hoeffding / Hoeffding-Adaptive / **Adaptive RF(最强)** / knn

### 3. Feature-based(资源分组)MoE(新架构,Module 4-5)
- 按资源类型分 expert(CPU / Memory / I/O / Network),gate 做**加权融合(soft routing)**
- 学习型 gate(非负最小二乘权重)+ importance gate(消融)
- 资源重要性分析(CPU 主导 ~93%)
- 诚实结论:在 CPU 主导数据上 resource-MoE ≈ single(workload-MoE 仍最强);价值需在更均衡负载上验证

### 4. 能耗感知 Offloading 决策(Objective 3)
- 能耗优先的贪心 knapsack:把高能耗 job offload 到更绿色、容量受限的 secondary 集群
- 5 策略对比:all_primary / random / single / moe / oracle
- 实测:MoE 节能 6.2%,弥合 single→oracle 决策差距的 87%

### 5. 真实 Snakemake 调度集成
- 真实 Snakemake DAG:`plan` → 每 job 一个 `run_job` → `aggregate`
- 决策逻辑与仿真器共用同一 `select_offloaded`
- 诚实边界:两个集群在单机模拟(忙循环缩放),非真实多物理集群

### 6. 部署:静态推理 + Live Online Learning + 闭环
- **静态推理**(`serve`):`/predict` only,模型不变
- **Live online API**(`serve-online`):`/predict` + `/update`,手动 curl 反馈
- **Online workflow 闭环**(`online-workflow`):job runner 自动逐 job `/predict`(dispatch 前)+ `/update`(完成后),**无需手动 curl**;state 持久化、重启可恢复
- 已验证是**真 Online-MoE**(gate + 4 个独立 expert,更新一个不影响其他三个)

---

## 项目结构

```
energy-offloading/
├── moe/                    # Workload-grouped MoE + online learning(Task 3/4/5)
│   ├── data.py             # 加载 work/,按 interval 聚合
│   ├── registry.py         # 模型注册表(batch 7 个 + online 6 个)+ 在线能力映射
│   ├── moe.py              # MixtureOfExperts(gate+experts)+ SingleModel
│   ├── run_moe_baseline.py # Task 4:MoE vs Single(5 折 CV / --time-split)
│   ├── run_online.py       # Task 5:静态 vs 在线 MoE 抗漂移
│   ├── compare_models.py   # 全模型调查
│   └── online_baseline_comparison.py  # 7 家族统一对比表
├── feature_moe/            # Feature-based(资源分组)MoE(新架构)
│   ├── groups.py           # 特征→资源组(CPU/Mem/IO/Net)
│   ├── importance.py       # 资源重要性(permutation)
│   ├── moe.py              # ResourceMoE(4 expert + learned gate)
│   └── run_resource_moe.py # 评估 + 对比
├── offloading/             # 能耗感知 offloading 决策引擎
│   ├── clusters.py         # primary/secondary 集群模型
│   ├── decision.py         # 能耗优先 knapsack(共用 select_offloaded)
│   ├── workflow.py         # 真实 interval→job 切分
│   ├── run_offloading.py   # 仿真运行器
│   └── online_workflow.py  # 闭环驱动:自动 /predict + /update
├── snakemake_integration/  # 真实 Snakemake DAG
├── inference/              # 推理服务(静态 + 在线)
│   ├── server.py           # serve:静态 HTTP
│   ├── online_server.py    # serve-online:live online learning HTTP
│   ├── predictor.py / online_predictor.py
├── moe_export/             # 导出可部署 artifact
├── deploy/                 # smoke test 脚本 + K8s 模板
├── results/                # 实验结果(文本/图/CSV)
├── Dockerfile / docker-compose.yml / docker-entrypoint.sh
└── DOCKER.md               # Docker 完整使用文档
```

---

## 怎么用

### 用 Docker(推荐,一键复现)

```sh
# 1. 构建
cd energy-offloading
docker build -t energy-offloading:latest .

# 2. 跑任意任务(挂载数据)
DATA=/home/hujiao/MPDS/work

docker run --rm -v $DATA:/data/work:ro energy-offloading:latest task4 --expert linear
docker run --rm -v $DATA:/data/work:ro energy-offloading:latest task5
docker run --rm -v $DATA:/data/work:ro energy-offloading:latest offload --expert linear
docker run --rm -v $DATA:/data/work:ro energy-offloading:latest snakemake compare
docker run --rm -v $DATA:/data/work:ro energy-offloading:latest online       # 7 家族对比表

# 新架构 resource-MoE(绕过 entrypoint)
docker run --rm --entrypoint bash -v $DATA:/data/work:ro energy-offloading:latest \
  -lc "PYTHONPATH=. python -m feature_moe.run_resource_moe --expert rf"
```

#### 子命令一览

| 命令 | 作用 |
|---|---|
| `serve` | 静态推理 HTTP 服务(port 8800) |
| `serve-online` | **Live online learning** 服务(`/predict` + `/update`) |
| `online-workflow` | **闭环**:job runner 自动 `/predict` + `/update` |
| `task4` | MoE vs Single Model(5 折 CV) |
| `task5` | 静态 vs 在线 MoE 抗漂移(ARF) |
| `online` | 7 家族 online/offline 统一对比表 |
| `compare` | 全 batch+online 模型调查 |
| `offload` | 能耗感知 offloading 仿真 |
| `snakemake` | 真实 Snakemake offloading DAG |
| `export` / `export-online` | 重新导出模型 artifact |
| `all` | task4 + task5 + online + offload |

### Live Online Learning 闭环(核心演示)

```sh
# 起 online 服务,state 持久化到挂载卷
mkdir -p ./online_state
docker run -d --name energy-online -p 8800:8800 \
  -e ONLINE_STATE_PATH=/app/models/state/online_state.pkl \
  -v "$PWD/online_state:/app/models/state" \
  energy-offloading:latest serve-online

# 手动闭环
curl -s -X POST localhost:8800/predict -H 'Content-Type: application/json' \
  -d '{"features":{"delta_cpu_ns":5e8,"delta_instructions":2e9}}'
# -> {"prediction_id":"...","energy_wh":268.19,"expert":"DAW1","model_version":0}

# job 完成后回传真实能耗 → 模型增量更新
curl -s -X POST localhost:8800/update -H 'Content-Type: application/json' \
  -d '{"prediction_id":"<上面返回的id>","true_energy_wh":250.0}'
# -> {"updated_expert":"DAW1","num_updates":1,"model_version":1}

# 自动闭环(无需手动 curl):逐 job predict→执行→update
docker run --rm -v $DATA:/data/work:ro \
  -v "$PWD/online_state:/app/models/state" \
  -v "$PWD/results:/app/results" \
  energy-offloading:latest online-workflow --jobs-per-workload 5 \
  --out /app/results/online_workflow_run.json
```

### 不用 Docker(直接 venv)

```sh
cd energy-offloading
PYTHONPATH=. /home/hujiao/MPDS/.venv/bin/python -m moe.run_moe_baseline --expert linear
PYTHONPATH=. /home/hujiao/MPDS/.venv/bin/python -m feature_moe.run_resource_moe --expert rf
# ...其余同理,把 `docker run ... <cmd>` 换成 `PYTHONPATH=. python -m ...`
```

### 一键验证(smoke test)

```sh
./deploy/smoke_test.sh                 # 静态推理
./deploy/online_smoke_test.sh          # 在线 API(predict→update→重启持久化)
./deploy/online_workflow_smoke_test.sh # 闭环(自动 predict+update+重启)
```

---

## 关键实验结果(已落盘 `results/`)

| 实验 | 结果 |
|---|---|
| Task 4(linear) | MoE R² 0.888 vs single 0.792(+0.10) |
| Task 5(ARF online) | Online-MoE R² 0.93 / MAE 6.2 vs 静态 0.71 / 18.3 |
| 7 家族对比 | RF/HGB 0.964(离线上限),Adaptive RF 0.936(最强在线) |
| Offloading | MoE 节能 6.2%,弥合 87% single→oracle 差距 |
| Resource-MoE(RF) | 0.960 ≈ single(CPU 主导,收益有限) |

---

## 诚实边界

- **Offloading / 多集群**:仿真,无真实第二物理集群;集群参数是合理假设
- **在线闭环的 `true_energy_wh`**:来自实测 workload 数据(模拟 job 完成信号),非运行时实时测量;接真实 Snakemake hook / cluster monitor 是剩余生产步骤
- **K8s manifest**(`deploy/k8s/`):可用模板,**未真正 apply**(环境无集群)
- **Resource-MoE**:实现且 sound,但在 CPU 主导数据上不增益,workload-MoE 仍最强
- **别人的 `ProcessEnergyAccounting/` 仓库**:全程未改动,仅读其 `work/` 数据

完整部署/使用细节见 **`DOCKER.md`**;各模块细节见对应子目录的 `README.md`。
