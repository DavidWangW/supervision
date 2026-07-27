# 高速公路视觉分析与事故风险预警系统 — 实施路线图

> 本文档记录系统的现状评估、差距分析与分阶段实施规划，供后续迭代调整参考。
> 最后更新：2026-07-24

---

## 一、现状评估

当前 `examples/supervision-service` 是一个**离线视频分析系统**，架构清晰、工程质量高。

| 层 | 技术栈 | 已实现能力 |
|---|---|---|
| 前端 | Vue3 + TS + Vite | 视频上传/预览、四点标定交互、任务列表、历史记录 |
| 后端 | FastAPI + BackgroundTasks | 异步任务提交(202)、H.264 转码、GPU 批量推理 |
| 存储 | SQLite (`uploads`/`processing_jobs`) | 仅存**任务元数据**，不存结构化分析结果 |
| AI | YOLO + ByteTrack + 透视变换 | ① 检测+跟踪(中文标注) ② 四点标定测速 |

### 关键特征与局限

1. **处理模式是"文件→文件"**：上传视频 → 后台跑完 → 输出标注后的 MP4。**没有实时流**。
2. **结果只落在渲染视频里**：车速/ID 只是画在画面上，**没有结构化数据入库**，无法做后续统计与预测。
3. **车型只有 COCO 4 类**（car/motorcycle/bus/truck），无法区分 SUV/货车/卡车等细粒度。
4. **只有交通流感知的一半**：有测速、有跟踪，缺车流量、车头间距、天气/路面、风险预测。

---

## 二、目标与差距分析

最终目标拆为 **3 大支柱 + 1 个决策层**：

```
支柱1 交通流感知   支柱2 环境感知      支柱3 风险预测
车速 ✅            天气识别 ❌         事故概率模型 ❌
车头间距 ❌        路面状况 ❌         风险等级 ❌
车道级流量 ❌      传感器接入 ❌            ↓
细粒度车型 ❌                        决策层: 分钟级封闭/开启建议 ❌
跟踪 ✅
─────────────────────────────────────────────
底座缺口: 实时流接入 ❌ + 结构化数据管道 ❌（最关键）
```

**最核心的底座缺口是"结构化数据管道"**：没有它，测出来的速度、流量、天气都无法沉淀成可分析、可训练、可预测的数据资产。这是从"炫技 Demo"走向"生产系统"的分水岭。

---

## 三、分阶段实施路线图

分阶段推进，每阶段都有可交付、可演示成果，且后一阶段依赖前一阶段的数据积累。

### 阶段 0：数据底座重构（最高优先级，1–2 周）

> 不产出新功能，但决定整个系统天花板。**先把"数据"跑通，再谈智能。**

1. **抽象统一分析管道 `AnalysisPipeline`**
   把 `video_processor.py` / `speed_processor.py` 中重复的"取帧→检测→跟踪→标注→写帧"逻辑抽象成可插拔管道，检测/跟踪只做一次，各分析器（测速、流量、间距、车型）作为**插件**挂上去。避免每加一个功能就复制一份处理循环。

2. **扩展数据库 Schema**（SQLite 先用，量大后迁 PostgreSQL/TimescaleDB）：
   - `vehicle_tracks`：每辆车的轨迹（track_id, class, 首末时间, 平均/最大速度, 所在车道）
   - `traffic_metrics`：按分钟聚合（时间戳, 车道, 流量, 平均车速, 平均车头间距, 各车型占比, 平均密度）
   - `environment_readings`：天气/路面状况时间序列
   - `accident_events`：历史事故（时间, 位置, 类型, 当时天气/路面/流量快照）— 训练风险模型的标签源
   - `risk_assessments`：风险评估输出（时间, 风险等级, 概率, 关键因子）

3. **引入 `CSVSink`/`JSONSink` + 数据库双写**，让每次处理都沉淀结构化数据。

**交付**：一次视频处理后，数据库里能查到"第 3 分钟第 2 车道通过 42 辆车，均速 88km/h"。

---

### 阶段 1：交通流感知补全（2–3 周）

在统一管道上补齐三个分析器：

1. **车道级车流量** — 用 `sv.LineZone`（每车道一条计数线）或 `sv.PolygonZone`（车道多边形）。前端复用现有"标定交互"，让用户画车道线/车道多边形。
2. **车头间距 / 车头时距(THW)** — 复用测速的透视变换，把同车道车辆按行驶方向排序，算相邻车 BEV 坐标距离；结合速度得到时距（秒）。THW < 2s 触发预警。
3. **密度与占有率** — 由 PolygonZone 内车辆数 / 车道长度得到，是风险预测的关键特征。

**交付**：仪表盘实时展示每车道流量、均速、平均车头间距、密度。

---

### 阶段 2：细粒度车型分类（3–4 周，需数据标注）

COCO 无法区分 SUV/轿车/货车/卡车，两条路线：

- **路线 A（推荐先做）**：在 COCO 4 类基础上做**层级细化**——用车辆 bbox 宽高比、面积、纵横比启发式 + 轻量分类头，先把 truck 细分为"货车/卡车/挂车"。快速见效。
- **路线 B（长期）**：**自建数据集微调 YOLO**。采集本地摄像头图像，按目标类别（摩托车/小轿车/SUV/货车/卡车…）标注，用 `supervision` 的 `DetectionDataset` 管理 COCO/YOLO 格式，训练自定义权重。这是达标的正解。

**交付**：`traffic_metrics` 中各车型占比准确落库；货车占比是重要风险因子。

---

### 阶段 3：实时流接入（2–3 周，架构升级）✅ 已实现

从"文件处理"升级为"实时流处理"，且**与离线视频文件共用同一套分析引擎**：

1. **RTSP 接入**：`app/services/stream_manager.py` 中每路流一个常驻后台线程，用 OpenCV `cv2.VideoCapture(rtsp_url)` 消费实时流，连接断开时**指数退避自动重连**。
2. **背压处理**：每帧都解码以排空缓冲（降低时延），但仅在 `sample_fps` 采样率下跑推理，兼顾算力与时延。
3. **常驻分析进程**：`StreamManager` 单例持有 worker 注册表，负责 start/stop/inspect，并在应用关闭时 `stop_all()`。每个 worker 复用 `TrafficFrameAnalyzer`（离线/实时同一代码）。
4. **实时推送**：每路流暴露 MJPEG `StreamingResponse`（`/mjpeg`）用于带标注的实时画面，以及 WebSocket（`/ws`）每 0.4s 推送指标快照；前端另以 `/snapshot` 轮询作为降级通道。
5. **存储复用**：实时流复用 `traffic_metrics`/`risk_assessments` 表，以 stream id 作为 `upload_id`（按分钟分桶 flush），并新增 `stream_sources` 表保存配置。

**交付**：在「实时流」页输入 RTSP 地址，四点标定后即可实时滚动显示带标注画面、各车道流量/均速/时距/密度/货车占比与分钟级风险研判。离线文件模式完全不受影响。

---

### 阶段 4：环境感知（天气 + 路面）（3–4 周）✅ 已实现（业务/启发式版）

1. **基于画面的天气/路面分类**：`app/services/environment.py` 的 `SceneClassifier` 用经典 CV 启发式（亮度/对比度、暗通道先验测雾、拉普拉斯方差测对比、雨纹高频残差、雪白像素占比、路面镜面高光占比、昼夜判定）实时判别 `晴/阴/雨/雪/雾` 与路面 `干燥/潮湿/积水/结冰/积雪` 及能见度。接口已预留可扩展点：后续直接换成训练好的图像分类模型即可（detector 暂用 `yolo26x.pt` 占位，识别暂不走检测，走帧级统计特征）。
2. **传感器融合（可选）**：预留接口接入气象站/路面结冰传感器（温度、湿度、路面温度、能见度），与视觉结果做加权融合，提升鲁棒性。
3. 结果写入 `environment_readings`（新增 `details_json`/`model` 列），作为风险模型输入；实时流每完成一分钟自动落库一次自动识别结果，离线分析整体落库一次。

**交付**：仪表盘显示"当前：小雪 / 路面积雪 / 能见度中"，并入库；Analyze 页支持「识别当前帧」预览与一键采纳为人工标注，Stream 页实时滚动展示环境识别卡。

---

### 阶段 5：事故风险预测与决策支持（4–6 周，项目核心价值）

把前面所有数据变现的关键：

1. **特征工程**：从 `traffic_metrics` + `environment_readings` 构造特征向量（流量、均速、速度方差、密度、车头时距分布、货车占比、天气、路面、能见度、时段…）。
2. **建模**（数据量决定方法）：
   - 冷启动：**规则/评分卡**（专家规则加权，如"积雪+密度高+速度方差大 → 高风险"），可立即上线。
   - 数据积累后：**梯度提升树（XGBoost/LightGBM）** 做二分类/回归，输出事故概率。历史 `accident_events` 做正样本。
   - 长期：时序模型（LSTM/TCN）捕捉趋势。
3. **风险分级**：概率 → 4 级（低/中/高/极高），每级对应管理动作建议（正常/限速/间歇管控/封闭）。
4. **决策看板**：分钟级风险曲线 + 关键因子归因（可解释性，让管理部门信服）+ 管控建议 + 预警推送。

**交付**：管理端看到"未来 30 分钟 K12+500 段风险等级：高(概率 0.72)，主因：路面结冰+车头时距偏小，建议限速 60 并加密巡查"。

---

## 四、目标技术架构（升级后）

```
摄像头(RTSP)─┐                          ┌─ WebSocket/SSE ─ 前端仪表盘(实时)
气象/路面传感器─┼─▶ 采集/抽帧 ─▶ 统一分析管道 ─┼─ 结构化数据 ─▶ 时序库(TimescaleDB)
              │   (每路一个worker)   ├检测+跟踪                    │
              │                     ├测速/间距/流量/车型          ▼
              │                     └天气/路面分类          特征工程+风险模型
              │                                                   │
              └────────────────────────────────────────── 决策看板/预警/管控建议
```

### 技术选型演进

| 维度 | 现状 | 演进目标 |
|---|---|---|
| 存储 | SQLite | PostgreSQL + TimescaleDB（时序数据天然契合） |
| 任务 | BackgroundTasks | Celery/Redis 或独立 worker 进程（实时流常驻） |
| 实时通信 | 轮询 job 状态 | WebSocket |
| 风险模型 | 无 | 规则评分卡 → LightGBM → 时序深度模型 |
| 部署 | 单任务 | 多路摄像头 + GPU 调度（一卡多路） |

---

## 五、优先级建议

务实推进顺序：

1. **先做阶段 0（数据底座）+ 阶段 1（流量/间距）** —— 性价比最高，让系统从"演示"变成"能产出数据资产"，且完全基于现有 supervision 能力，无需训练模型。
2. **再做阶段 5 的规则评分卡版本** —— 哪怕数据不多，先用专家规则跑通"感知→风险→建议"的完整闭环，是核心卖点和 Demo 亮点。
3. **车型细分、天气识别、ML 风险模型**这些需要数据/训练的，与数据积累并行推进。

---

## 六、落地切入点（已实现 ✅）

已按 **A → B → C → D** 顺序实现：阶段 0（数据底座）+ 阶段 1（流量/间距）+ 阶段 5（规则评分卡）+ 阶段 3（实时流接入，架构升级）+ 阶段 4（环境感知，启发式识别版）。

### 已新增/修改文件

**后端**
- `app/db/database.py`：新增 `traffic_metrics` / `vehicle_tracks` / `environment_readings` / `accident_events` / `risk_assessments` 五张表。
- `app/db/analytics_repository.py`：结构化分析数据的读写层。
- `app/services/risk_engine.py`：规则评分卡风险引擎（天气/路面/车头时距/密度/货车占比 → 0–100 分 + 4 级 + 概率 + 因子解释）。
- `app/services/analytics.py`：统一分析管道 `TrafficFrameAnalyzer`（离线文件与实时流共用的逐帧分析逻辑，一次检测+跟踪，插件式产出车速/车道流量/车头间距/密度并写库）。
- `app/routers/analyze.py`：综合任务提交接口 `POST /api/v1/videos/analyze`。
- `app/routers/analytics.py`：分析结果查询接口（traffic-metrics / vehicle-tracks / risk / environment）。
- `app/services/job_runner.py` / `app/main.py`：接入 analyze 任务与新路由。

**阶段 3 实时流（新增）**
- `app/db/database.py`：新增 `stream_sources` 表（RTSP 地址、四点标定、车道数、环境、采样率、阈值、状态）。
- `app/db/stream_repository.py`：视频流源 CRUD（含删除时级联清理 analytics）。
- `app/db/analytics_repository.py`：新增 `latest_risk_assessments(upload_id)` 用于实时风险卡。
- `app/services/stream_manager.py`：`StreamManager` 单例 + 常驻 worker（CV2 重连退避、背压、MJPEG 帧、WebSocket 快照、按分钟 flush）。
- `app/routers/streams.py`：`/api/v1/streams` 路由（preview-frame / create / list / start / stop / delete / snapshot / mjpeg / ws）。
- `app/main.py`：注册 streams 路由，lifespan 关闭时 `stream_manager.stop_all()`。
- `webapp/vite.config.ts`：`/api` 代理增加 `ws: true` 支持 WebSocket。

**阶段 4 环境感知（新增）**
- `app/services/environment.py`：`SceneClassifier` 帧级视觉环境识别（天气/路面/能见度，经典 CV 启发式 + 可扩展接口，便于后续替换为训练模型）。
- `app/db/database.py`：`environment_readings` 增加 `details_json`（完整识别结果）/ `model`（识别器标识）列迁移。
- `app/db/analytics_repository.py`：`insert_environment_reading` 支持 `details`/`model`；新增 `latest_environment_reading(upload_id)`。
- `app/services/analytics.py`：`TrafficFrameAnalyzer` 集成逐帧环境识别（指数平滑特征 + `aggregate_environment` 稳定整段结果），写入 `LiveSnapshot` 并参与风险评分。
- `app/services/stream_manager.py`：实时流每分钟自动落库自动识别结果，并以识别结果驱动风险评分。
- `app/routers/environment.py`：新增 `/api/v1/environment/detect`（base64 帧识别）+ `/api/v1/environment/labels`（可选标签）路由。
- `app/routers/analytics.py`：新增 `/api/v1/analytics/environment/latest` 最近环境识别接口。
- `app/main.py`：注册 environment 路由。
- `webapp/src/api/environment.ts`：环境识别 API 客户端。
- `webapp/src/components/EnvBars.vue`：环境识别概率条可复用组件。
- `webapp/src/views/AnalyzeView.vue`：增加「识别当前帧」预览 + 采纳为人工标注 + 结果区环境卡。
- `webapp/src/views/StreamView.vue`：实时面板增加「环境识别（实时）」卡。

**前端**
- `webapp/src/api/analytics.ts`：分析结果 API 客户端。
- `webapp/src/api/streams.ts`：实时流 API 客户端（预览帧、CRUD、MJPEG/WS URL）。
- `webapp/src/views/AnalyzeView.vue`：综合分析页面（标定 + 车道数 + 环境 + 风险卡 + 指标表 + 轨迹表）。
- `webapp/src/views/StreamView.vue`：实时流监控页（新建/标定、列表管理、MJPEG 实时画面、WebSocket 实时指标与风险卡）。
- `webapp/src/router/index.ts` + `AppLayout.vue`：新增 `/stream` 路由与「实时流」导航项。

### 待办（后续阶段）
- 阶段 2 细粒度车型（YOLO 微调）、阶段 4 的训练模型替换（将 `SceneClassifier` 启发式换成图像分类模型）、阶段 5 的 ML 风险模型（XGBoost/LightGBM）。
