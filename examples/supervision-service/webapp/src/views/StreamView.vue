<script setup lang="ts">
import { computed, onBeforeUnmount, ref, watch } from 'vue'

import CalibrationCanvas from '@/components/CalibrationCanvas.vue'
import EnvBars from '@/components/EnvBars.vue'
import type { EnvProb } from '@/api/environment'
import type { LiveRisk, LiveSnapshot, SourcePoint, StreamRecord } from '@/api/streams'
import {
  createStream,
  deleteStream,
  fetchPreviewFrame,
  listStreams,
  startStream,
  stopStream,
  streamMjpegUrl,
  streamWsUrl,
} from '@/api/streams'

type FormPhase = 'idle' | 'preview' | 'calibrate'

const streams = ref<StreamRecord[]>([])
const errorMessage = ref('')
const loadError = ref('')

// --- Create form state ---
const formPhase = ref<FormPhase>('idle')
const formName = ref('')
const formUrl = ref('')
const previewImage = ref('')
const previewWidth = ref(0)
const previewHeight = ref(0)
const sourcePoints = ref<SourcePoint[]>([])
const targetWidth = ref(25)
const targetHeight = ref(250)
const laneCount = ref(3)
const weather = ref('')
const roadCondition = ref('')
const visibility = ref('')
const sampleFps = ref(10)
const confidenceThreshold = ref(0.3)
const iouThreshold = ref(0.7)
const isFetchingPreview = ref(false)
const isCreating = ref(false)

// --- Live monitor state ---
const liveStreamId = ref('')
const liveSnapshot = ref<LiveSnapshot | null>(null)
const liveStatus = ref('')
const liveFps = ref(0)

let socket: WebSocket | null = null
let pollTimer: ReturnType<typeof setInterval> | null = null

const WEATHER_OPTIONS = ['', '晴', '多云', '雨', '雪', '雾', '霾']
const ROAD_OPTIONS = ['', '干燥', '潮湿', '积水', '积雪', '结冰']
const VISIBILITY_OPTIONS = ['', '好', '中', '差']

const liveRisk = computed<LiveRisk | null>(() => liveSnapshot.value?.risk ?? null)
const liveOverall = computed(() => liveRisk.value?.overall ?? null)
const liveLaneRisks = computed(() => liveRisk.value?.lanes ?? [])
const liveEnv = computed<EnvProb | null>(() => liveSnapshot.value?.environment ?? null)
const liveVlmActive = computed<boolean>(() => liveSnapshot.value?.vlm_active ?? true)
const liveEnvModel = computed<string>(() => {
  const model = liveEnv.value?.model ?? ''
  return model.startsWith('vlm:') ? 'IntelliVisionEnv' : model
})

const liveMjpegSrc = computed(() =>
  liveStreamId.value && isLiveRunning.value ? streamMjpegUrl(liveStreamId.value) : '',
)
const isLiveRunning = computed(() => liveStatus.value === 'running')
const liveStream = computed(() => streams.value.find((s) => s.id === liveStreamId.value))

const riskColor: Record<string, string> = {
  低: 'var(--success)',
  中: 'var(--marking)',
  高: '#ff8a3d',
  极高: 'var(--danger)',
}

const statusMeta: Record<string, { label: string; color: string }> = {
  running: { label: '运行中', color: 'var(--success)' },
  starting: { label: '启动中', color: 'var(--marking)' },
  reconnecting: { label: '重连中', color: '#ff8a3d' },
  stopped: { label: '已停止', color: 'var(--text-muted)' },
  error: { label: '错误', color: 'var(--danger)' },
}

function statusInfo(status: string): { label: string; color: string } {
  return (
    statusMeta[status] ??
    statusMeta.stopped ?? { label: '未知', color: 'var(--text-muted)' }
  )
}

async function loadStreams() {
  loadError.value = ''
  try {
    streams.value = await listStreams()
  } catch (error) {
    loadError.value = error instanceof Error ? error.message : '加载视频流列表失败。'
  }
}

function fmt(v: number | null | undefined, digits = 1): string {
  return v === null || v === undefined ? '—' : v.toFixed(digits)
}

function fmtPct(v: number | null | undefined): string {
  return v === null || v === undefined ? '—' : `${(v * 100).toFixed(0)}%`
}

// --- Create form handlers ---
async function onGetPreview() {
  if (!formUrl.value.trim()) {
    errorMessage.value = '请先填写视频流地址。'
    return
  }
  errorMessage.value = ''
  isFetchingPreview.value = true
  sourcePoints.value = []
  previewImage.value = ''
  try {
    const frame = await fetchPreviewFrame(formUrl.value.trim())
    previewImage.value = frame.image
    previewWidth.value = frame.width
    previewHeight.value = frame.height
    formPhase.value = 'calibrate'
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '获取预览帧失败。'
    formPhase.value = 'idle'
  } finally {
    isFetchingPreview.value = false
  }
}

function onPointsUpdate(points: SourcePoint[]) {
  sourcePoints.value = points
}

function resetForm() {
  formPhase.value = 'idle'
  formName.value = ''
  formUrl.value = ''
  previewImage.value = ''
  sourcePoints.value = []
  errorMessage.value = ''
}

async function onCreate() {
  if (!formUrl.value.trim()) {
    errorMessage.value = '请填写视频流地址。'
    return
  }
  if (sourcePoints.value.length !== 4) {
    errorMessage.value = '请在预览画面上标记四个路面角点。'
    return
  }
  if (!formName.value.trim()) {
    errorMessage.value = '请填写视频流名称。'
    return
  }
  errorMessage.value = ''
  isCreating.value = true
  try {
    const created = await createStream({
      name: formName.value.trim(),
      url: formUrl.value.trim(),
      sourcePoints: sourcePoints.value,
      targetWidth: targetWidth.value,
      targetHeight: targetHeight.value,
      laneCount: laneCount.value,
      weather: weather.value || undefined,
      roadCondition: roadCondition.value || undefined,
      visibility: visibility.value || undefined,
      sampleFps: sampleFps.value,
      confidenceThreshold: confidenceThreshold.value,
      iouThreshold: iouThreshold.value,
    })
    resetForm()
    await loadStreams()
    openLive(created.id)
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '创建视频流失败。'
  } finally {
    isCreating.value = false
  }
}

// --- Stream control handlers ---
async function onStart(id: string) {
  try {
    await startStream(id)
    await loadStreams()
    if (liveStreamId.value === id) {
      openLive(id)
    }
  } catch (error) {
    loadError.value = error instanceof Error ? error.message : '启动失败。'
  }
}

async function onStop(id: string) {
  try {
    await stopStream(id)
    await loadStreams()
    if (liveStreamId.value === id) {
      liveStatus.value = 'stopped'
      liveSnapshot.value = null
    }
  } catch (error) {
    loadError.value = error instanceof Error ? error.message : '停止失败。'
  }
}

async function onDelete(id: string) {
  if (!confirm('确认删除该视频流及其全部分析数据？')) {
    return
  }
  try {
    await deleteStream(id)
    if (liveStreamId.value === id) {
      closeLive()
    }
    await loadStreams()
  } catch (error) {
    loadError.value = error instanceof Error ? error.message : '删除失败。'
  }
}

// --- Live monitor (WebSocket + polling fallback) ---
function handleSnapshot(data: string) {
  try {
    const payload = JSON.parse(data) as {
      type: string
      status?: string
      fps_actual?: number
      snapshot?: LiveSnapshot
    }
    if (payload.type === 'snapshot') {
      liveStatus.value = payload.status ?? liveStatus.value
      liveFps.value = payload.fps_actual ?? 0
      liveSnapshot.value = payload.snapshot ?? null
    } else if (payload.type === 'stopped') {
      liveStatus.value = payload.status ?? 'stopped'
      liveSnapshot.value = null
      closeSocket()
    } else if (payload.type === 'error') {
      liveStatus.value = 'error'
      liveSnapshot.value = null
      closeSocket()
    }
  } catch {
    // ignore malformed frames
  }
}

function openLive(id: string) {
  closeLive()
  liveStreamId.value = id
  liveSnapshot.value = null
  liveStatus.value = 'starting'
  liveFps.value = 0
  try {
    socket = new WebSocket(streamWsUrl(id))
    let opened = false
    socket.onopen = () => {
      opened = true
    }
    socket.onmessage = (event) => handleSnapshot(event.data as string)
    socket.onerror = () => {
      if (!opened) {
        startPolling(id)
      }
    }
    socket.onclose = () => {
      socket = null
    }
  } catch {
    startPolling(id)
  }
}

function startPolling(id: string) {
  stopPolling()
  pollTimer = setInterval(async () => {
    try {
      const res = await fetch(`/api/v1/streams/${id}/snapshot`)
      if (!res.ok) {
        return
      }
      const data = (await res.json()) as { status: string; snapshot: LiveSnapshot | null }
      liveStatus.value = data.status
      liveSnapshot.value = data.snapshot
    } catch {
      // transient; keep polling
    }
  }, 1000)
}

function stopPolling() {
  if (pollTimer) {
    clearInterval(pollTimer)
    pollTimer = null
  }
}

function closeSocket() {
  if (socket) {
    try {
      socket.close()
    } catch {
      // ignore
    }
    socket = null
  }
}

function closeLive() {
  closeSocket()
  stopPolling()
  liveStreamId.value = ''
  liveSnapshot.value = null
  liveStatus.value = ''
  liveFps.value = 0
}

watch(
  () => streams.value.map((s) => s.id).join(','),
  () => {
    if (liveStreamId.value && !streams.value.some((s) => s.id === liveStreamId.value)) {
      closeLive()
    }
  },
)

onBeforeUnmount(() => {
  closeLive()
})
</script>

<template>
  <div class="workspace page">
    <header class="page-header">
      <div>
        <p class="eyebrow">Real-time Module</p>
        <h1>实时视频流监控</h1>
        <p class="lede">
          接入 RTSP 视频流进行实时检测、跟踪与风险研判，与离线视频文件共用同一套分析引擎。
        </p>
      </div>
    </header>

    <p v-if="loadError" class="error banner">{{ loadError }}</p>

    <div class="layout">
      <section class="main-col">
        <div class="panel">
          <header class="panel-head">
            <h2>新建视频流</h2>
            <span class="muted">RTSP / 视频流地址</span>
          </header>

          <div class="create-grid">
            <div class="create-feed">
              <div class="url-row">
                <input
                  v-model="formUrl"
                  class="url-input"
                  type="text"
                  placeholder="rtsp://localhost:8554/stream1"
                  @keyup.enter="onGetPreview"
                />
                <button type="button" class="ghost" :disabled="isFetchingPreview" @click="onGetPreview">
                  {{ isFetchingPreview ? '获取中…' : '获取预览帧' }}
                </button>
              </div>

              <CalibrationCanvas
                v-if="previewImage && formPhase === 'calibrate'"
                :frame-url="previewImage"
                :video-width="previewWidth"
                :video-height="previewHeight"
                @update="onPointsUpdate"
              />
              <div v-else class="feed-placeholder">
                <p>填写视频流地址后点击「获取预览帧」，在画面中标记四个路面角点完成标定。</p>
              </div>
            </div>

            <aside class="create-controls">
              <label class="field">
                <span>名称</span>
                <input v-model="formName" type="text" placeholder="例如：匝道相机 A" />
              </label>
              <label class="field">
                <span>车道数量</span>
                <input v-model.number="laneCount" type="number" min="1" max="8" step="1" />
              </label>
              <label class="field">
                <span>路面宽度 (m)</span>
                <input v-model.number="targetWidth" type="number" min="1" step="1" />
              </label>
              <label class="field">
                <span>路面长度 (m)</span>
                <input v-model.number="targetHeight" type="number" min="1" step="1" />
              </label>
              <label class="field">
                <span>采样帧率 (fps)</span>
                <input v-model.number="sampleFps" type="number" min="1" max="30" step="1" />
              </label>
              <label class="field">
                <span>天气</span>
                <select v-model="weather">
                  <option v-for="w in WEATHER_OPTIONS" :key="w" :value="w">{{ w || '未知' }}</option>
                </select>
              </label>
              <label class="field">
                <span>路面</span>
                <select v-model="roadCondition">
                  <option v-for="r in ROAD_OPTIONS" :key="r" :value="r">{{ r || '未知' }}</option>
                </select>
              </label>
              <label class="field">
                <span>能见度</span>
                <select v-model="visibility">
                  <option v-for="v in VISIBILITY_OPTIONS" :key="v" :value="v">{{ v || '未知' }}</option>
                </select>
              </label>
              <p class="hint small">天气/路面/能见度留空时，系统将基于实时画面自动识别并参与风险评分。</p>
              <label class="field">
                <span>置信度 {{ confidenceThreshold.toFixed(2) }}</span>
                <input v-model.number="confidenceThreshold" type="range" min="0" max="1" step="0.05" />
              </label>
              <label class="field">
                <span>IOU {{ iouThreshold.toFixed(2) }}</span>
                <input v-model.number="iouThreshold" type="range" min="0" max="1" step="0.05" />
              </label>
              <dl class="stats">
                <div>
                  <dt>标定点</dt>
                  <dd>{{ sourcePoints.length }} / 4</dd>
                </div>
              </dl>
              <button
                type="button"
                class="primary"
                :disabled="isCreating || sourcePoints.length !== 4"
                @click="onCreate"
              >
                {{ isCreating ? '创建中…' : '创建并启动' }}
              </button>
            </aside>
          </div>
          <p v-if="errorMessage" class="error">{{ errorMessage }}</p>
        </div>

        <div class="panel">
          <header class="panel-head">
            <h2>已配置视频流</h2>
            <button type="button" class="ghost small" @click="loadStreams">刷新</button>
          </header>
          <div v-if="streams.length === 0" class="empty">
            <p>暂无视频流，使用上方表单接入一个 RTSP 源。</p>
          </div>
          <ul v-else class="stream-list">
            <li v-for="s in streams" :key="s.id" class="stream-card">
              <div class="stream-info">
                <div class="stream-title">
                  <strong>{{ s.name }}</strong>
                  <span
                    class="badge"
                    :style="{ color: statusInfo(s.status).color }"
                  >
                    {{ statusInfo(s.status).label }}
                  </span>
                </div>
                <code class="stream-url">{{ s.url }}</code>
                <div class="stream-meta">
                  <span>车道 {{ s.lane_count }}</span>
                  <span v-if="s.runtime"> · {{ s.runtime.fps_actual }} fps</span>
                  <span v-if="s.runtime"> · {{ s.runtime.frames_processed }} 帧</span>
                </div>
                <p v-if="s.error_message" class="stream-error">{{ s.error_message }}</p>
              </div>
              <div class="stream-actions">
                <button
                  type="button"
                  class="ghost small"
                  :disabled="s.running"
                  @click="onStart(s.id)"
                >
                  启动
                </button>
                <button
                  type="button"
                  class="ghost small"
                  :disabled="!s.running"
                  @click="onStop(s.id)"
                >
                  停止
                </button>
                <button
                  type="button"
                  class="accent small"
                  :disabled="!s.running"
                  @click="openLive(s.id)"
                >
                  监控
                </button>
                <button type="button" class="danger small" @click="onDelete(s.id)">删除</button>
              </div>
            </li>
          </ul>
        </div>
      </section>

      <aside class="live-col">
        <div class="panel live-panel">
          <header class="panel-head">
            <h2>实时监控</h2>
            <button v-if="liveStreamId" type="button" class="ghost small" @click="closeLive">
              关闭
            </button>
          </header>

          <div v-if="!liveStreamId" class="live-placeholder">
            <p>从左侧列表选择一个视频流，点击「监控」查看实时画面与指标。</p>
          </div>

          <template v-else>
            <p v-if="liveStream" class="live-name">
              {{ liveStream.name }}
              <span
                class="badge"
                :style="{ color: statusInfo(liveStatus).color }"
              >
                {{ statusInfo(liveStatus).label }}
              </span>
            </p>

            <img v-if="liveMjpegSrc" :src="liveMjpegSrc" class="live-feed" alt="实时画面" />
            <div v-else class="live-feed off">画面不可用（视频流未运行）</div>

            <article
              v-if="liveOverall"
              class="risk-card"
              :style="{ borderColor: riskColor[liveOverall.risk_level] ?? 'var(--line)' }"
            >
              <p class="risk-eyebrow">整体风险等级</p>
              <h3
                class="risk-level"
                :style="{ color: riskColor[liveOverall.risk_level] ?? 'var(--text)' }"
              >
                {{ liveOverall.risk_level }}
              </h3>
              <p class="risk-score">
                评分 {{ liveOverall.risk_score }} · 事故概率
                {{ liveOverall.probability != null ? (liveOverall.probability * 100).toFixed(0) + '%' : '—' }}
              </p>
              <ul class="factors">
                <li v-for="(f, i) in liveOverall.factors ?? []" :key="i">{{ f }}</li>
              </ul>
            </article>
            <div v-else class="risk-card muted-card">
              <p class="risk-eyebrow">整体风险等级</p>
              <h3 class="risk-level">—</h3>
              <p class="risk-score">等待数据…</p>
            </div>

            <div v-if="liveLaneRisks.length" class="lane-risks">
              <article
                v-for="r in liveLaneRisks"
                :key="r.lane_id"
                class="lane-risk"
                :style="{ borderColor: riskColor[r.risk_level] }"
              >
                <span class="lane-tag">车道 {{ (r.lane_id ?? 0) + 1 }}</span>
                <strong :style="{ color: riskColor[r.risk_level] }">{{ r.risk_level }}</strong>
                <span class="lane-score">评分 {{ r.risk_score }}</span>
              </article>
            </div>

            <div v-if="liveSnapshot" class="live-stats">
              <div><span>总车辆</span><b>{{ liveSnapshot.total_vehicles }}</b></div>
              <div><span>实际帧率</span><b>{{ liveFps }} fps</b></div>
              <div><span>当前分钟</span><b>{{ liveSnapshot.minute_bucket }}</b></div>
            </div>

            <div v-if="liveEnv" class="panel env-live">
              <header class="panel-head">
                <h3>环境识别（实时）</h3>
                <span class="muted">{{ liveEnvModel }}</span>
              </header>
              <p v-if="!liveVlmActive" class="env-paused-hint">
                IntelliVisionEnv环境识别已暂停
              </p>
              <EnvBars :env="liveEnv" />
            </div>

            <div v-if="liveSnapshot && liveSnapshot.lanes.length" class="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>车道</th>
                    <th>流量</th>
                    <th>均速(km/h)</th>
                    <th>最小时距(s)</th>
                    <th>密度</th>
                    <th>货车占比</th>
                  </tr>
                </thead>
                <tbody>
                  <tr v-for="lane in liveSnapshot.lanes" :key="lane.lane_id">
                    <td>{{ lane.lane_id + 1 }}</td>
                    <td>{{ lane.flow_total }}</td>
                    <td>{{ fmt(lane.avg_speed_kmh) }}</td>
                    <td>{{ fmt(lane.min_headway_s, 2) }}</td>
                    <td>{{ fmt(lane.density, 0) }} 辆/km</td>
                    <td>{{ fmtPct(lane.truck_ratio) }}</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </template>
        </div>
      </aside>
    </div>
  </div>
</template>

<style scoped>
.page {
  padding: 2rem clamp(1rem, 4vw, 2.5rem) 3rem;
}

.page-header {
  margin-bottom: 1.5rem;
}

.eyebrow {
  margin: 0 0 0.5rem;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.72rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--marking);
}

.page-header h1 {
  margin: 0 0 0.5rem;
  font-family: 'Barlow Condensed', sans-serif;
  font-size: clamp(1.8rem, 3vw, 2.4rem);
}

.lede {
  margin: 0;
  color: var(--text-muted);
}

.error.banner {
  border: 1px solid var(--danger);
  padding: 0.6rem 0.85rem;
  margin-bottom: 1rem;
}

.layout {
  display: grid;
  grid-template-columns: minmax(0, 1.4fr) minmax(320px, 0.9fr);
  gap: 1rem;
  align-items: start;
}

.main-col {
  display: grid;
  gap: 1rem;
}

.panel {
  border: 1px solid var(--line);
  background: var(--bg-panel);
  padding: 1rem;
}

.panel-head {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  margin-bottom: 0.85rem;
}

.panel-head h2 {
  margin: 0;
  font-family: 'Barlow Condensed', sans-serif;
  font-size: 1.1rem;
  letter-spacing: 0.05em;
}

.muted {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.75rem;
  color: var(--text-muted);
}

.create-grid {
  display: grid;
  grid-template-columns: minmax(0, 1.5fr) minmax(220px, 0.9fr);
  gap: 1rem;
  align-items: start;
}

.create-feed {
  display: grid;
  gap: 0.75rem;
}

.url-row {
  display: flex;
  gap: 0.5rem;
}

.url-input {
  flex: 1;
  padding: 0.6rem 0.7rem;
  border: 1px solid rgba(139, 148, 158, 0.45);
  background: var(--bg-base);
  color: var(--text);
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.85rem;
}

.feed-placeholder {
  border: 1px dashed rgba(139, 148, 158, 0.4);
  padding: 2rem 1rem;
  text-align: center;
  color: var(--text-muted);
  font-size: 0.85rem;
}

.create-controls {
  display: grid;
  gap: 0.75rem;
}

.field {
  display: grid;
  gap: 0.3rem;
}

.field span {
  font-size: 0.8rem;
  color: var(--text-muted);
}

.field input,
.field select {
  padding: 0.5rem 0.6rem;
  border: 1px solid rgba(139, 148, 158, 0.45);
  background: var(--bg-base);
  color: var(--text);
  font-family: 'JetBrains Mono', monospace;
}

.hint {
  margin: 0 0 0.75rem;
  color: var(--text-muted);
  font-size: 0.82rem;
}

.hint.small {
  margin: 0 0 0.25rem;
  font-size: 0.76rem;
}

.env-live {
  margin-top: 0.75rem;
  display: grid;
  gap: 0.5rem;
}

.env-paused-hint {
  margin: 0;
  padding: 0.4rem 0.6rem;
  border-radius: 6px;
  background: rgba(232, 163, 61, 0.12);
  border: 1px solid rgba(232, 163, 61, 0.4);
  color: #e8a33d;
  font-size: 0.78rem;
  font-family: 'JetBrains Mono', monospace;
  letter-spacing: 0.02em;
}

.stats {
  margin: 0;
  display: grid;
  gap: 0.5rem;
}

.stats div {
  display: flex;
  justify-content: space-between;
  padding-bottom: 0.4rem;
  border-bottom: 1px solid rgba(139, 148, 158, 0.2);
}

.stats dt {
  font-size: 0.78rem;
  color: var(--text-muted);
}

.stats dd {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.82rem;
  color: var(--marking);
}

.primary,
.ghost,
.accent,
.danger {
  padding: 0.7rem 1rem;
  border: 0;
  cursor: pointer;
  font-family: 'Barlow Condensed', sans-serif;
  font-size: 1rem;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}

.primary {
  background: var(--accent);
  color: var(--bg-base);
}

.primary:disabled {
  opacity: 0.45;
  cursor: not-allowed;
}

.ghost {
  background: transparent;
  border: 1px solid rgba(139, 148, 158, 0.45);
  color: var(--text-muted);
}

.ghost.small,
.accent.small,
.danger.small {
  padding: 0.4rem 0.7rem;
  font-size: 0.82rem;
}

.accent {
  background: rgba(255, 107, 44, 0.12);
  border: 1px solid rgba(255, 107, 44, 0.45);
  color: var(--marking);
}

.danger {
  background: transparent;
  border: 1px solid var(--danger);
  color: var(--danger);
}

.error {
  margin: 0.85rem 0 0;
  color: var(--danger);
  font-size: 0.875rem;
}

.empty {
  color: var(--text-muted);
  font-size: 0.88rem;
  padding: 0.5rem 0;
}

.stream-list {
  list-style: none;
  margin: 0;
  padding: 0;
  display: grid;
  gap: 0.65rem;
}

.stream-card {
  display: flex;
  justify-content: space-between;
  gap: 1rem;
  padding: 0.85rem;
  border: 1px solid rgba(139, 148, 158, 0.22);
}

.stream-title {
  display: flex;
  align-items: center;
  gap: 0.6rem;
}

.stream-title strong {
  font-family: 'Barlow Condensed', sans-serif;
  font-size: 1.05rem;
}

.badge {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.72rem;
  letter-spacing: 0.05em;
}

.stream-url {
  display: block;
  margin: 0.3rem 0;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.78rem;
  color: var(--text-muted);
  word-break: break-all;
}

.stream-meta {
  font-size: 0.78rem;
  color: var(--text-muted);
}

.stream-error {
  margin: 0.35rem 0 0;
  color: var(--danger);
  font-size: 0.78rem;
}

.stream-actions {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0.4rem;
  align-content: start;
}

.live-panel {
  position: sticky;
  top: 1rem;
}

.live-placeholder {
  color: var(--text-muted);
  font-size: 0.88rem;
  padding: 2rem 0;
  text-align: center;
}

.live-name {
  margin: 0 0 0.6rem;
  font-family: 'Barlow Condensed', sans-serif;
  font-size: 1.1rem;
  display: flex;
  align-items: center;
  gap: 0.6rem;
}

.live-feed {
  width: 100%;
  display: block;
  border: 1px solid rgba(139, 148, 158, 0.25);
  background: #000;
  min-height: 180px;
}

.live-feed.off {
  color: var(--text-muted);
  text-align: center;
  padding: 3rem 1rem;
  font-size: 0.85rem;
}

.risk-card {
  margin-top: 0.85rem;
  border: 1px solid var(--line);
  background: var(--bg-panel);
  padding: 1rem;
}

.muted-card {
  opacity: 0.7;
}

.risk-eyebrow {
  margin: 0 0 0.3rem;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.7rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--text-muted);
}

.risk-level {
  margin: 0 0 0.35rem;
  font-family: 'Barlow Condensed', sans-serif;
  font-size: 2.4rem;
  line-height: 1;
}

.risk-score {
  margin: 0 0 0.6rem;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.82rem;
  color: var(--text-muted);
}

.factors {
  margin: 0;
  padding-left: 1.1rem;
  display: grid;
  gap: 0.3rem;
  font-size: 0.82rem;
}

.lane-risks {
  margin-top: 0.65rem;
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(110px, 1fr));
  gap: 0.5rem;
}

.lane-risk {
  border: 1px solid var(--line);
  padding: 0.6rem;
  display: grid;
  gap: 0.2rem;
}

.lane-tag {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.7rem;
  color: var(--text-muted);
}

.lane-risk strong {
  font-family: 'Barlow Condensed', sans-serif;
  font-size: 1.2rem;
}

.lane-score {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.74rem;
  color: var(--text-muted);
}

.live-stats {
  margin-top: 0.85rem;
  display: grid;
  gap: 0.45rem;
}

.live-stats div {
  display: flex;
  justify-content: space-between;
  padding-bottom: 0.4rem;
  border-bottom: 1px solid rgba(139, 148, 158, 0.2);
}

.live-stats span {
  font-size: 0.8rem;
  color: var(--text-muted);
}

.live-stats b {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.85rem;
  color: var(--marking);
}

.table-wrap {
  margin-top: 0.85rem;
  max-height: 280px;
  overflow: auto;
  border: 1px solid rgba(139, 148, 158, 0.2);
}

table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.78rem;
}

th,
td {
  padding: 0.4rem 0.5rem;
  text-align: left;
  border-bottom: 1px solid rgba(139, 148, 158, 0.15);
  white-space: nowrap;
}

th {
  position: sticky;
  top: 0;
  background: var(--bg-panel);
  color: var(--text-muted);
  font-weight: 600;
}

@media (max-width: 980px) {
  .layout,
  .create-grid {
    grid-template-columns: 1fr;
  }

  .live-panel {
    position: static;
  }
}
</style>
