# 高速公路视觉分析与事故风险预警系统 — 进度与路线图（续作交接文档）

> 用途：本文档供**新会话**接续开发使用。包含项目目标、当前进度、运行方式、已实现能力、API 清单、已知问题与修复、未提交改动风险、细化路线图与下一步行动清单。
> 最后更新：**2026-07-28**
> 配套详细规划见同目录 `ROADMAP.md`（分阶段路线图）。本文偏“如何接续”，ROADMAP 偏“为什么这么做”。

---

## 一、项目目标（明确）

把 `supervision` 视频分析库扩展成一个**高速公路场景的生产级视觉分析 + 事故风险预警系统**，超越原来的“上传视频→出标注视频”Demo，落地为能沉淀结构化数据、可实时、可风险研判的系统。

最终形态（3 大支柱 + 1 个决策层）：

```
支柱1 交通流感知   支柱2 环境感知      支柱3 风险预测
车速 ✅            天气/路面/能见度 ✅  事故概率模型 ❌(规则卡✅)
车头间距 ✅        传感器融合 ❌         风险等级 ✅(规则评分卡)
车道级流量 ✅      视觉识别模型 ❌(启发式✅)  ↓
细粒度车型 ❌                        决策层: 分钟级管控建议 ✅(规则)
跟踪 ✅
─────────────────────────────────────────────
底座: 实时流接入 ✅ + 结构化数据管道 ✅（已完成）
```

当前**已实现**：阶段 0（数据底座）+ 阶段 1（流量/间距）+ 阶段 3（实时流接入）+ 阶段 4（环境感知·启发式版）+ 阶段 5 的规则评分卡部分。
**尚未做**：阶段 2（细粒度车型 YOLO 微调）、阶段 4 的“训练模型替换启发式”、阶段 5 的 ML 风险模型（XGBoost/LightGBM）、传感器融合。

**当前模型权重约定**：detector 统一用 `models/yolo26x.pt`（已存在）。`models/` 目录下另含 `yolo11x.pt` / `yolo26m.pt` / `yolov8s.pt` 可作替换（改 `app/config.py` 的 `DEFAULT_WEIGHTS` / `DEFAULT_SPEED_WEIGHTS`）。环境识别**不走检测**，走帧级经典 CV 统计特征（`SceneClassifier`）。训练好的专用模型后续替换 `SceneClassifier` 内部即可，对外接口不变。

---

## 二、系统架构概览

```
摄像头(RTSP) ─┐
上传视频文件  ─┼─▶ 统一分析管道 TrafficFrameAnalyzer ─▶ 结构化数据(CSV/Sink + SQLite)
             │   (离线文件 与 实时流 共用同一套逐帧逻辑)
             │      ├ 检测+跟踪(sv/YOLO+ByteTrack)
             │      ├ 测速/车道流量/车头间距/密度(透视变换)
             │      ├ 环境识别(SceneClassifier, 逐帧)
             │      └ 风险评分卡(risk_engine)
             └─▶ 实时推送: MJPEG(/mjpeg) + WebSocket(/ws) ─▶ Vue3 前端仪表盘
```

- **后端**：FastAPI（`app/main.py` 为入口，lifespan 启动时 `init_db()`、关闭时 `stream_manager.stop_all()`）。
- **前端**：Vue 3 + TS + Vite（`webapp/`）。
- **存储**：SQLite（`data/supervision.db`），含 `uploads` / `processing_jobs` / `traffic_metrics` / `vehicle_tracks` / `environment_readings` / `accident_events` / `risk_assessments` / `stream_sources`。
- **核心复用点**：`TrafficFrameAnalyzer`（在 `app/services/analytics.py`）同时服务离线视频与实时流，做到“一次检测+跟踪，多分析器插件式产出”。

---

## 三、当前进度（已交付）

| 阶段 | 状态 | 关键交付 |
|---|---|---|
| 阶段 0 数据底座 | ✅ 已实现 | 5 张结构化表 + `analytics_repository` + 双写；`traffic_metrics`/`vehicle_tracks` 按分钟/轨迹落库 |
| 阶段 1 交通流感知 | ✅ 已实现 | 车道级流量、车头间距(THW)、密度、货车占比；`TrafficFrameAnalyzer` 统一管道 |
| 阶段 3 实时流接入 | ✅ 已实现 | `stream_manager` 常驻 worker（RTSP 重连退避、背压、MJPEG、WS、按分钟 flush）；`stream_sources` 表；前端实时页 |
| 阶段 4 环境感知 | ✅ 已实现（启发式版） | `SceneClassifier` 帧级识别天气/路面/能见度 + 概率；`/environment/detect`、`/environment/latest`；前端识别卡 |
| 阶段 5 风险决策 | 🟡 部分（规则卡） | `risk_engine` 规则评分卡（天气/路面/车头时距/密度/货车占比→0–100 分+4 级+概率+因子解释）；决策看板 UI |
| 阶段 2 细粒度车型 | ❌ 未做 | 待 YOLO 微调 |
| 阶段 4 训练模型 | ❌ 未做 | 待替换为图像分类模型 |
| 阶段 5 ML 风险模型 | ❌ 未做 | 待 XGBoost/LightGBM |

---

## 四、运行方式（接续必读）

### 环境
- Python >= 3.10；Node.js >= 22.18；NVIDIA GPU 可选（CPU 可跑但慢）。
- 使用 **uv** 管理 venv（项目 venv 在 `D:\MyGithubRepo\supervision\.venv`）。
- **本地 `supervision` 引用已自动处理**：`app/__init__.py` 在包导入时把仓库 `src/` 前置到 `sys.path`，因此无论用 `uv run` 还是系统/IDE 解释器，`import supervision` 都指向**当前仓库**版本（修复了此前解析到 PyPI 的问题，见第七节 8）。推荐 IDE 解释器仍指向本服务 `.venv` 以保证依赖完整。
- 权重 `models/yolo26x.pt` 已存在。首次运行 ultralytics 会自动下载/复用。

### 启动（开发模式，推荐）
```powershell
# 终端 1 — 后端（端口 8005，API 文档 http://127.0.0.1:8005/docs）
cd examples/supervision-service
uv run D:/MyGithubRepo/supervision/.venv/Scripts/python.exe -m uvicorn app.main:app --port 8005 --reload

# 终端 2 — 前端（端口 5175，代理 /api 到 8005）
cd examples/supervision-service/webapp
npm run dev
```
- Web 控制台：http://127.0.0.1:5175
- 生产模式：`npm run build` 后 `fastapi run app/main.py`（单端口 8000）。

### 已安装的额外依赖（需注意复现）
- **PyAV (`av`)**：用于 RTSP H.265 解码回退。已写入 `pyproject.toml`，但当前 venv 是早期用 `uv pip install --python .../python.exe av` 装的；新 venv 用 `uv sync` 会按 pyproject 自动带上。
- 当前 venv 没有 `pip`，用 `uv pip` 安装包。

---

## 五、API 清单（已实现端点）

### 视频 / 分析
| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/v1/videos/analyze` | 综合任务提交（标定+车道数+环境→跑分析+落库） |
| GET | `/api/v1/analytics/traffic-metrics` | 交通指标（按 upload_id） |
| GET | `/api/v1/analytics/vehicle-tracks` | 车辆轨迹 |
| GET | `/api/v1/analytics/risk` | 风险评估 |
| GET | `/api/v1/analytics/environment` | 环境读数列表 |
| GET | `/api/v1/analytics/environment/latest` | 最近（自动识别）环境读数 |
| POST | `/api/v1/videos/track` | 检测+跟踪，返回 MP4 |
| POST | `/api/v1/videos/speed-estimate` | 速度估算，返回 MP4 |
| GET/POST | `/api/v1/videos/...` | 上传/预览/服务端样例视频 |

### 实时流 `streams`
| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/v1/streams/preview-frame` | 取一帧用于标定/预览 |
| POST | `/api/v1/streams` | 新建流源 |
| GET | `/api/v1/streams` | 列表 |
| GET | `/api/v1/streams/{id}` | 详情 |
| POST | `/api/v1/streams/{id}/start` | 启动 worker |
| POST | `/api/v1/streams/{id}/stop` | 停止 |
| DELETE | `/api/v1/streams/{id}` | 删除（级联清理 analytics） |
| GET | `/api/v1/streams/{id}/snapshot` | 当前标注帧（轮询降级通道） |
| GET | `/api/v1/streams/{id}/mjpeg` | 带标注实时画面（MJPEG） |
| WS | `/api/v1/streams/{id}/ws` | 每 ~0.4s 推送指标快照（含 `environment`） |

### 环境识别 `environment`
| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/v1/environment/detect` | 传 base64 帧 → 返回天气/路面/能见度 + 概率 + 特征 |
| GET | `/api/v1/environment/labels` | 候选标签集（天气/路面/能见度/模型名） |

### 其它
- `GET /health` 健康检查；`GET /api/info` 元信息；`records` 路由（上传/任务记录 CRUD）。

---

## 六、关键文件索引（改动都在这）

**后端**
- `app/services/analytics.py` — `TrafficFrameAnalyzer` 统一逐帧分析管道（流量/间距/密度 + 环境识别钩子 + `aggregate_environment`）。
- `app/services/environment.py` — `SceneClassifier` 帧级视觉环境识别（经典 CV 启发式 + 可扩展接口）。**替换训练模型只改这里。**
- `app/services/risk_engine.py` — 规则评分卡风险引擎。
- `app/services/stream_manager.py` — `StreamManager` 单例 + 常驻 worker（**RTSP 多传输方式探测 + PyAV 回退**在此）。
- `app/services/job_runner.py` — 离线任务执行。
- `app/db/database.py` — 建表 + 列迁移（`environment_readings` 的 `details_json`/`model`）。
- `app/db/analytics_repository.py` — 结构化读写（`insert_environment_reading` 支持 details/model；`latest_environment_reading`）。
- `app/db/stream_repository.py` — 流源 CRUD。
- `app/routers/*` — analytics / analyze / environment / streams / records / videos / speed / health。
- `app/config.py` — 路径、扩展名、`DEFAULT_WEIGHTS=models/yolo26x.pt`、批处理大小。

**前端**
- `webapp/src/api/analytics.ts` / `streams.ts` / `environment.ts` — API 客户端。
- `webapp/src/components/EnvBars.vue` — 环境识别概率条组件。
- `webapp/src/views/AnalyzeView.vue` — 综合分析页（标定+车道+环境+风险卡+指标/轨迹表；“识别当前帧”+采纳为人工标注）。
- `webapp/src/views/StreamView.vue` — 实时流监控页（MJPEG+WS 实时指标+环境识别卡）。
- `webapp/src/router/index.ts` + `AppLayout.vue` — 路由与导航。

---

## 七、已知问题与修复记录（避免重踩）

1. **RTSP 能 VLC 播放但 OpenCV 读不到帧**
   - 根因：OpenCV 自带 ffmpeg 局限（强制 TCP 不匹配源 / 不支持 H.265）。
   - 修复：`stream_manager.py` 改为 **TCP→UDP→默认** 三传输方式探测 + **PyAV 回退**（`_connect_source` / `_try_opencv` / `_try_pyav` / `FrameSource` 抽象）。
   - 依赖：已装 `av`（PyAV）。新 venv 用 `uv sync` 会自动装（已写入 pyproject）。
   - 若仍失败：看后端日志 `Stream connected via <opencv|pyav>/<tcp|udp|default>` 判断命中哪条路径。

2. **`cv2.Laplacian` 运行时崩溃**
   - 根因：`gray` 是 `float32`(格式5)，`cv2.Laplacian(..., cv2.CV_64F)` 触发 AVX2 不支持 float32→float64。
   - 修复：`environment.py` 中改用 `cv2.CV_32F`。合成帧测试通过。

3. **venv 无 `pip`**
   - 用 `uv pip install --python <venv>/Scripts/python.exe <pkg>` 安装。

4. **中文日志在 Windows cmd 下乱码**
   - 仅显示问题，不影响功能；PowerShell / 写文件查看即可。

### 2026-07-28 会话修复（运行期关键 Bug，已全部解决 ✅）

> 以下 4 个问题均已在 `develop` 工作树修复并**已提交**，新会话可直接接续，无需重做。

5. **点击「创建并启动」无画面，报 `TypeError: '_thread.lock' object is not callable`**
   - 根因：`_StreamShared` 的 `lock` 字段写成 `field(default_factory=threading.Lock())`，
     在**类定义时**就把锁实例当工厂调用，导致 `lock` 被赋值为「调用后的返回值」而非锁本身。
   - 修复：`stream_manager.py` 改为 `field(default_factory=threading.Lock)`（传类，而非实例）。

6. **有画面但无检测框 / 无车道线 / 无风险信息**
   - 根因：`_run` 创建了 `infer_thread` 推理线程，却**漏写 `.start()`**，推理线程从未运行，
     导致 `shared.analyzer` 始终为 `None`，`process()` 不产生任何标注/指标。
   - 修复：`_run` 中补上 `infer_thread.start()`。

7. **检测框与实时监控画面不同步、无法跟踪车辆**
   - 根因：显示线程把「帧 N 的缓存检测框」用 `annotate_cached(frame)` 重画到**最新帧**上，
     而 `process()` 本返回「原帧画框」的完整标注帧却被丢弃 → 框永远滞后、不跟随车辆。
   - 修复：
     - 推理线程改为发布**推理产出的标注帧**（`latest_annotated` + 单调递增 `annot_seq`），
       显示线程仅在 `annot_seq` 变化时重推，保证「框与产生它的原始帧同源」。
     - `analytics.py` 场景分类器**降频**到每 6 帧（`fi % 6 == 0`）跑一次，其余帧复用上一结果，
       推理吞吐↑、框率↑；并加 **MJPEG 保活**（`_KEEPALIVE_DT=0.5s`）在推理慢时重推同帧防前端超时。
     - 清理了 `analytics.py` 中已无用的 `_last_detections`/`_last_labels`/`_annot_lock` 与 `annotate_cached` 死代码。

8. **`analytics.py` 中 `import supervision as sv` 报错（解析到 PyPI 版而非本项目版）**
   - 根因：IDE / 系统 Python 未激活 editable 安装，Python 优先解析到 PyPI 上的 `supervision`，
     而非仓库 `src/supervision`（后者含本项目定制/依赖的特性）。
   - 修复：在 `app/__init__.py` 包导入时**前置本地 `src` 到 `sys.path`**（必须在任何 `import supervision` 之前）：
     ```python
     import sys
     from pathlib import Path
     _REPO_ROOT = Path(__file__).resolve().parents[3]
     _LOCAL_SV_SRC = str(_REPO_ROOT / "src")
     if _LOCAL_SV_SRC not in sys.path:
         sys.path.insert(0, _LOCAL_SV_SRC)
     ```
   - 影响：无论用 `uv run` venv 还是系统/IDE 解释器启动，都引用**当前仓库**的 `supervision`。
     建议 IDE 运行/调试的解释器仍指向本服务 `.venv` 以保证依赖完整。

9. **「开始综合分析」报错：`Both input arrays must be (arrays of) 3-dimensional vectors, but they are 2 and 2 dimensional instead.`**
   - 根因：**不是 OpenCV**，而是 **NumPy 2.x 改变了 `np.cross` 行为**——现在要求交叉轴长度为 3（三维向量），但本仓库 `src/supervision` 里两处用了二维向量：
     - `src/supervision/detection/utils/internal.py` 的 `cross_product()`（被 `sv.LineZone` 调用，用于车道流量计数，在 `analytics.py` 的 `process()` 第 464 行 `self.line_counters[lane_id].trigger(in_det)` 触发）；
     - `src/supervision/geometry/utils.py` 的 `get_polygon_center()`（多边形质心，shoelace 公式）。
   - 修复：两处都改为把二维顶点 **补零成三维 (x, y, 0)** 后做 `np.cross`，再取 z 分量（等价于原二维叉积的标量结果），对 NumPy 1.x/2.x 均兼容。
   - 验证：`analyzer.process()` 跑通；`sv.get_polygon_center` 返回正确质心。
   - ⚠️ 此修复改动了**仓库根目录的 `src/supervision`（本地库）**，不在 `examples/supervision-service` 子目录内，提交时请连同根仓库一起 `git add`。

---

## 八、代码提交状态（2026-07-28 更新）

**截至本次同步，`examples/supervision-service` 已全部纳入 git 跟踪并提交（约 99 个文件，`git status --short examples/supervision-service` 无未提交改动）。** 阶段 0/1/3/4 + 规则风险卡 + 上述 4 个运行期修复均已在 `develop` 工作树中。

> ⚠️ 仍需留意：
> - 这些提交**尚未推送到 `origin/develop`** 前，仅在本地。需要协同/备份时执行：
>   ```powershell
>   cd d:\MyGithubRepo\supervision
>   git push origin develop
>   ```
> - 仍**不要**对任意目录执行 `git clean -fdx`（会误删 `uploads/`、`outputs/`、`models/` 等未跟踪但重要的运行产物与权重）。
> - 根仓库另有未跟踪项（`解决方案/`、`*.docx`、`temp_read.txt`），与本项目无关，勿一并提交。

## 八·补充、当前现存问题（结余，未解决）

### 已解决（运行期，见第七节 5–8）
- 无画面（`threading.Lock` 工厂）、无检测框（漏 `infer_thread.start()`）、框不同步/不跟踪（发布推理标注帧 + 场景分类器降频 + MJPEG 保活）、`import supervision` 指向 PyPI（本地 `src` 前置 `sys.path`）。

### 仍待解决（架构 / 数据 / 模型层面，非运行期阻断）
1. **阶段 2 细粒度车型**：当前 COCO 4 类（car/motorcycle/bus/truck），无 SUV/货车/卡车细分；需自建数据集微调 YOLO（用 `supervision` 的 `DetectionDataset`）。
2. **阶段 4 训练模型替换**：`SceneClassifier` 目前是经典 CV 启发式；接口（`classify()`/`classify_features()`/`to_dict()`）已预留，直接换图像分类模型即可，前端零改动。
3. **阶段 5 ML 风险模型**：当前是规则评分卡，待积累数据后上 XGBoost/LightGBM（历史 `accident_events` 作正样本）。
4. **传感器融合**：`SceneClassifier` 预留气象站/路面传感器接入点，未实现。
5. **实时流算力依赖 GPU**：CPU 下推理帧率低、框率受影响；RTSP H.265 需 PyAV 回退（已接，依赖 `av`）。
6. **存储/部署**：SQLite 单机、BackgroundTasks 轻量；多路摄像头 + 高并发需 PostgreSQL/TimescaleDB + Celery。

---

## 九、细化路线图（下一步做什么）

### 立即可做（低风险，价值高）
1. **提交当前工作**（见上）。
2. **端到端联调验证**：启动后端+前端，上传一段高速视频跑 `analyze`，确认 `traffic_metrics`/`risk_assessments`/`environment_readings` 落库；再用 `rtsp://192.168.131.128:8554/stream1` 走实时流。
3. **扩充 `SceneClassifier` 启发式**：当前是简单阈值，可加更稳健的特征（如天空占比、路面反光方向性、雾的暗通道方差），并补充单元测试。

### 中期（需数据/训练）
4. **阶段 4 训练模型替换**：采集/整理天气-路面图像数据集，训练图像分类模型（或复用现成道路天气分类权重），在 `SceneClassifier` 加 `if USE_TRAINED_MODEL: ... else: 启发式` 分支。接口（`detect`/`labels`）不变，前端零改动。
5. **阶段 2 细粒度车型**：用 `supervision` 的 `DetectionDataset` 管理本地标注，微调 YOLO；在 `TrafficFrameAnalyzer` 增加车型细分字段并落库（货车占比是重要风险因子）。
6. **阶段 5 ML 风险模型**：积累 `traffic_metrics`+`environment_readings` 后，做特征工程 → LightGBM/XGBoost 二分类（历史 `accident_events` 作正样本）→ 替换/补充规则卡，保留可解释性（因子归因）。

### 长期（架构演进）
7. **传感器融合**：`SceneClassifier` 预留接口接入气象站/路面结冰传感器，与视觉结果加权融合。
8. **存储/部署升级**：SQLite→PostgreSQL+TimescaleDB；BackgroundTasks→Celery/独立 worker；多路摄像头+GPU 调度。

---

## 十、新会话接续清单（照做即可）

1. `cd examples/supervision-service` → 先 `git status` 确认未提交工作还在（见第八节）。
2. 启动后端（8005）+ 前端（5175），打开 http://127.0.0.1:5175。
3. 想继续**环境识别模型训练** → 改 `app/services/environment.py` 的 `SceneClassifier`，保持 `classify()`/`classify_features()`/`to_dict()` 接口。
4. 想继续**风险 ML 模型** → 改 `app/services/risk_engine.py`，新增模型分支，保持 `assess()` 返回 `RiskResult`。
5. 想继续**车型细分** → 在 `app/services/analytics.py` 的 `TrafficFrameAnalyzer` 扩展车型字段并同步 `analytics_repository` + 前端表格。
6. 任何后端改动后重启 uvicorn；前端 `npm run type-check` 保证类型通过。

---

## 附：验证命令（已通过，可作回归基线）
```powershell
# 后端导入
cd examples/supervision-service
uv run D:/MyGithubRepo/supervision/.venv/Scripts/python.exe -c "import app.main; print('APP_OK')"

# 分类器合成帧
uv run .../python.exe -c "import numpy as np; from app.services.environment import SceneClassifier; c=SceneClassifier(); print(c.classify(np.full((480,640,3),200,np.uint8)).weather)"

# 前端
cd webapp && npm run type-check && npm run build
```
