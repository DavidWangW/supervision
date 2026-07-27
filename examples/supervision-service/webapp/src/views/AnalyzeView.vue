<script setup lang="ts">
import { computed, onBeforeUnmount, ref } from 'vue'

import CalibrationCanvas from '@/components/CalibrationCanvas.vue'
import ProcessingState from '@/components/ProcessingState.vue'
import ResultPanel from '@/components/ResultPanel.vue'
import ServerVideoPicker from '@/components/ServerVideoPicker.vue'
import UploadProgress from '@/components/UploadProgress.vue'
import VideoDropzone from '@/components/VideoDropzone.vue'
import { uploadPreview } from '@/api/speed'
import type { ServerVideo } from '@/api/video'
import type { SourcePoint } from '@/api/speed'
import type { ProcessingJobRecord } from '@/api/records'
import { jobResultUrl } from '@/api/records'
import {
  analyzeTraffic,
  fetchJob,
  fetchRiskAssessments,
  fetchTrafficMetrics,
  fetchVehicleTracks,
  type RiskAssessment,
  type TrafficMetric,
  type VehicleTrack,
} from '@/api/analytics'
import {
  detectEnvironment,
  fetchLatestEnvironment,
  type EnvProb,
  type EnvironmentReading,
} from '@/api/environment'
import EnvBars from '@/components/EnvBars.vue'

type Phase = 'upload' | 'calibrate' | 'processing' | 'result'

const phase = ref<Phase>('upload')
const selectedFile = ref<File | null>(null)
const uploadId = ref('')
const selectedServer = ref('')
const previewUrl = ref('')
const frameUrl = ref('')
const videoWidth = ref(0)
const videoHeight = ref(0)
const sourcePoints = ref<SourcePoint[]>([])
const targetWidth = ref(25)
const targetHeight = ref(250)
const laneCount = ref(3)
const weather = ref('')
const roadCondition = ref('')
const visibility = ref('')
const confidenceThreshold = ref(0.3)
const iouThreshold = ref(0.7)
const errorMessage = ref('')
const resultVideoUrl = ref('')
const isUploading = ref(false)
const uploadProgress = ref(0)
const progress = ref(0)
const currentFrame = ref(0)
const totalFrames = ref(0)

const metrics = ref<TrafficMetric[]>([])
const tracks = ref<VehicleTrack[]>([])
const risks = ref<RiskAssessment[]>([])
const environment = ref<EnvironmentReading | null>(null)

// --- On-the-fly frame recognition (calibration preview) ---
const detectedPreview = ref<EnvProb | null>(null)
const detecting = ref(false)
const detectError = ref('')

let pollTimer: ReturnType<typeof setInterval> | null = null

const WEATHER_OPTIONS = ['', '晴', '多云', '雨', '雪', '雾', '霾']
const ROAD_OPTIONS = ['', '干燥', '潮湿', '积水', '积雪', '结冰']
const VISIBILITY_OPTIONS = ['', '好', '中', '差']

const overallRisk = computed(() => risks.value.find((r) => r.lane_id === null) ?? null)
const laneRisks = computed(() => risks.value.filter((r) => r.lane_id !== null))
const trackTypeCounts = computed(() => {
  const counts: Record<string, number> = {}
  for (const t of tracks.value) {
    const key = t.vehicle_type ?? '其他'
    counts[key] = (counts[key] ?? 0) + 1
  }
  return counts
})

const riskColor: Record<string, string> = {
  低: 'var(--success)',
  中: 'var(--marking)',
  高: '#ff8a3d',
  极高: 'var(--danger)',
}

function revokeFrameUrl() {
  if (frameUrl.value) {
    URL.revokeObjectURL(frameUrl.value)
    frameUrl.value = ''
  }
}

function captureFirstFrame(url: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const video = document.createElement('video')
    video.preload = 'metadata'
    video.src = url
    video.muted = true
    video.playsInline = true
    video.onloadeddata = () => {
      videoWidth.value = video.videoWidth
      videoHeight.value = video.videoHeight
      video.currentTime = 0
    }
    video.onseeked = () => {
      const canvas = document.createElement('canvas')
      canvas.width = video.videoWidth
      canvas.height = video.videoHeight
      const context = canvas.getContext('2d')
      if (!context) {
        reject(new Error('无法读取视频帧。'))
        return
      }
      context.drawImage(video, 0, 0)
      canvas.toBlob(
        (blob) => {
          if (!blob) {
            reject(new Error('无法生成预览帧。'))
            return
          }
          revokeFrameUrl()
          frameUrl.value = URL.createObjectURL(blob)
          resolve()
        },
        'image/jpeg',
        0.92,
      )
    }
    video.onerror = () => reject(new Error('视频无法播放，请更换格式。'))
  })
}

async function onSelect(file: File) {
  errorMessage.value = ''
  selectedFile.value = file
  uploadId.value = ''
  selectedServer.value = ''
  previewUrl.value = ''
  sourcePoints.value = []
  phase.value = 'upload'
  isUploading.value = true
  uploadProgress.value = 0
  try {
    const preview = await uploadPreview(file, (percent) => {
      uploadProgress.value = percent
    })
    uploadId.value = preview.uploadId
    previewUrl.value = preview.previewUrl
    await captureFirstFrame(preview.previewUrl)
    phase.value = 'calibrate'
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '读取视频失败。'
    selectedFile.value = null
  } finally {
    isUploading.value = false
  }
}

async function onSelectServer(video: ServerVideo) {
  errorMessage.value = ''
  revokeFrameUrl()
  selectedFile.value = null
  uploadId.value = ''
  selectedServer.value = video.name
  sourcePoints.value = []
  phase.value = 'upload'
  try {
    previewUrl.value = video.previewUrl
    await captureFirstFrame(video.previewUrl)
    phase.value = 'calibrate'
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '读取视频失败。'
    selectedServer.value = ''
  }
}

function onPointsUpdate(points: SourcePoint[]) {
  sourcePoints.value = points
}

function stopPolling() {
  if (pollTimer) {
    clearInterval(pollTimer)
    pollTimer = null
  }
}

async function pollUntilDone(jobId: string, upload: string) {
  stopPolling()
  pollTimer = setInterval(async () => {
    try {
      const job: ProcessingJobRecord = await fetchJob(jobId)
      progress.value = job.progress
      currentFrame.value = job.current_frame
      totalFrames.value = job.total_frames
      if (job.status === 'completed') {
        stopPolling()
        resultVideoUrl.value = jobResultUrl(jobId)
        await loadAnalytics(upload)
        phase.value = 'result'
      } else if (job.status === 'failed') {
        stopPolling()
        errorMessage.value = job.error_message ?? '分析失败。'
        phase.value = 'calibrate'
      }
    } catch (error) {
      errorMessage.value = error instanceof Error ? error.message : '轮询任务状态失败。'
    }
  }, 1500)
}

async function loadAnalytics(upload: string) {
  const [m, t, r, env] = await Promise.all([
    fetchTrafficMetrics(upload),
    fetchVehicleTracks(upload),
    fetchRiskAssessments(upload),
    fetchLatestEnvironment(upload).catch(() => null),
  ])
  metrics.value = m
  tracks.value = t
  risks.value = r
  environment.value = env
}

function frameUrlToDataUrl(url: string): Promise<string> {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.onload = () => {
      const canvas = document.createElement('canvas')
      canvas.width = img.naturalWidth
      canvas.height = img.naturalHeight
      const ctx = canvas.getContext('2d')
      if (!ctx) {
        reject(new Error('无法绘制帧。'))
        return
      }
      ctx.drawImage(img, 0, 0)
      resolve(canvas.toDataURL('image/jpeg', 0.9))
    }
    img.onerror = () => reject(new Error('无法加载预览帧。'))
    img.src = url
  })
}

async function detectCurrentFrame() {
  if (!frameUrl.value) return
  detecting.value = true
  detectError.value = ''
  detectedPreview.value = null
  try {
    const dataUrl = await frameUrlToDataUrl(frameUrl.value)
    detectedPreview.value = await detectEnvironment(dataUrl)
  } catch (error) {
    detectError.value = error instanceof Error ? error.message : '环境识别失败。'
  } finally {
    detecting.value = false
  }
}

function adoptEnvironment(env: EnvProb | null) {
  if (!env) return
  weather.value = env.weather
  roadCondition.value = env.road_condition
  visibility.value = env.visibility
}

async function runAnalysis() {
  if (sourcePoints.value.length !== 4) {
    errorMessage.value = '请在画面上标记四个路面角点。'
    return
  }
  phase.value = 'processing'
  errorMessage.value = ''
  progress.value = 0
  currentFrame.value = 0
  totalFrames.value = 0
  metrics.value = []
  tracks.value = []
  risks.value = []
  environment.value = null
  try {
    const result = await analyzeTraffic({
      uploadId: uploadId.value || undefined,
      file: selectedFile.value ?? undefined,
      serverVideo: selectedServer.value || undefined,
      sourcePoints: sourcePoints.value,
      targetWidth: targetWidth.value,
      targetHeight: targetHeight.value,
      laneCount: laneCount.value,
      weather: weather.value || undefined,
      roadCondition: roadCondition.value || undefined,
      visibility: visibility.value || undefined,
      confidenceThreshold: confidenceThreshold.value,
      iouThreshold: iouThreshold.value,
    })
    await pollUntilDone(result.job_id, result.upload_id)
  } catch (error) {
    phase.value = 'calibrate'
    errorMessage.value = error instanceof Error ? error.message : '综合分析提交失败。'
  }
}

function startOver() {
  stopPolling()
  revokeFrameUrl()
  selectedFile.value = null
  uploadId.value = ''
  selectedServer.value = ''
  previewUrl.value = ''
  sourcePoints.value = []
  resultVideoUrl.value = ''
  metrics.value = []
  tracks.value = []
  risks.value = []
  environment.value = null
  detectedPreview.value = null
  phase.value = 'upload'
  errorMessage.value = ''
}

function fmt(v: number | null, digits = 1): string {
  return v === null || v === undefined ? '—' : v.toFixed(digits)
}

function fmtPct(v: number | null): string {
  return v === null || v === undefined ? '—' : `${(v * 100).toFixed(0)}%`
}

onBeforeUnmount(() => {
  stopPolling()
  revokeFrameUrl()
})
</script>

<template>
  <div class="workspace page">
    <header class="page-header">
      <div>
        <p class="eyebrow">Analytics Module</p>
        <h1>综合交通分析</h1>
        <p class="lede">
          一次处理完成车速、车道流量、车头间距与密度的结构化分析，并基于规则评分卡给出事故风险等级。
        </p>
      </div>
    </header>

    <div class="grid">
      <section class="feed">
        <template v-if="phase === 'upload'">
          <UploadProgress v-if="isUploading" :progress="uploadProgress" />
          <template v-else>
            <ServerVideoPicker @select="onSelectServer" />
            <VideoDropzone @select="onSelect" />
          </template>
        </template>

        <CalibrationCanvas
          v-else-if="frameUrl && phase !== 'result'"
          :frame-url="frameUrl"
          :video-width="videoWidth"
          :video-height="videoHeight"
          @update="onPointsUpdate"
        />

        <ProcessingState
          v-if="phase === 'processing'"
          message="正在检测、跟踪并分析交通流与风险…"
          :progress="progress"
          :current-frame="currentFrame"
          :total-frames="totalFrames"
        />

        <ResultPanel
          v-if="phase === 'result' && resultVideoUrl"
          :url="resultVideoUrl"
          download-name="analyze_result.mp4"
          title="综合分析结果"
        />
      </section>

      <aside class="controls">
        <div class="block">
          <h2>标定与车道</h2>
          <p class="hint">四边形对应真实路面的宽与长（米），系统自动按车道数均分。</p>
          <label class="field">
            <span>路面宽度 (m)</span>
            <input v-model.number="targetWidth" type="number" min="1" step="1" />
          </label>
          <label class="field">
            <span>路面长度 (m)</span>
            <input v-model.number="targetHeight" type="number" min="1" step="1" />
          </label>
          <label class="field">
            <span>车道数量</span>
            <input v-model.number="laneCount" type="number" min="1" max="8" step="1" />
          </label>
        </div>

        <div class="block">
          <h2>环境状况（自动识别）</h2>
          <p class="hint">系统基于画面自动识别天气/路面/能见度并参与风险评分；如不准确可手动覆盖，或先“识别当前帧”预览。</p>
          <label class="field">
            <span>天气</span>
            <select v-model="weather">
              <option v-for="w in WEATHER_OPTIONS" :key="w" :value="w">
                {{ w || '未知' }}
              </option>
            </select>
          </label>
          <label class="field">
            <span>路面</span>
            <select v-model="roadCondition">
              <option v-for="r in ROAD_OPTIONS" :key="r" :value="r">
                {{ r || '未知' }}
              </option>
            </select>
          </label>
          <label class="field">
            <span>能见度</span>
            <select v-model="visibility">
              <option v-for="v in VISIBILITY_OPTIONS" :key="v" :value="v">
                {{ v || '未知' }}
              </option>
            </select>
          </label>
          <button
            type="button"
            class="ghost detect-btn"
            :disabled="!frameUrl || detecting"
            @click="detectCurrentFrame"
          >
            {{ detecting ? '识别中…' : '识别当前帧' }}
          </button>
          <p v-if="detectError" class="error">{{ detectError }}</p>
          <div v-if="detectedPreview" class="env-card">
            <p class="env-eyebrow">当前帧识别结果</p>
            <EnvBars :env="detectedPreview" />
            <button type="button" class="accent small" @click="adoptEnvironment(detectedPreview)">
              采纳为人工标注
            </button>
          </div>
        </div>

        <div class="block">
          <h2>检测阈值</h2>
          <label class="field">
            <span>置信度 {{ confidenceThreshold.toFixed(2) }}</span>
            <input v-model.number="confidenceThreshold" type="range" min="0" max="1" step="0.05" />
          </label>
          <label class="field">
            <span>IOU {{ iouThreshold.toFixed(2) }}</span>
            <input v-model.number="iouThreshold" type="range" min="0" max="1" step="0.05" />
          </label>
        </div>

        <dl class="stats">
          <div>
            <dt>分辨率</dt>
            <dd>{{ videoWidth && videoHeight ? `${videoWidth}×${videoHeight}` : '—' }}</dd>
          </div>
          <div>
            <dt>标定点</dt>
            <dd>{{ sourcePoints.length }} / 4</dd>
          </div>
        </dl>

        <div class="actions">
          <button
            type="button"
            class="primary"
            :disabled="phase === 'processing' || sourcePoints.length !== 4 || (!selectedFile && !selectedServer)"
            @click="runAnalysis"
          >
            {{ phase === 'processing' ? '处理中…' : '开始综合分析' }}
          </button>
          <button type="button" class="ghost" @click="startOver">换一段视频</button>
        </div>

        <p v-if="errorMessage" class="error">{{ errorMessage }}</p>
      </aside>
    </div>

    <section v-if="phase === 'result' && risks.length" class="results">
      <div class="result-grid">
        <article class="risk-card" :style="{ borderColor: overallRisk ? riskColor[overallRisk.risk_level] : 'var(--line)' }">
          <p class="risk-eyebrow">整体风险等级</p>
          <h2 class="risk-level" :style="{ color: overallRisk ? riskColor[overallRisk.risk_level] : 'var(--text)' }">
            {{ overallRisk?.risk_level ?? '—' }}
          </h2>
          <p class="risk-score">
            评分 {{ overallRisk?.risk_score ?? '—' }} · 事故概率
            {{ overallRisk?.probability != null ? (overallRisk.probability * 100).toFixed(0) + '%' : '—' }}
          </p>
          <ul class="factors">
            <li v-for="(f, i) in overallRisk?.factors ?? []" :key="i">{{ f }}</li>
          </ul>
        </article>

        <div class="lane-risks">
          <article
            v-for="r in laneRisks"
            :key="r.lane_id ?? 0"
            class="lane-risk"
            :style="{ borderColor: riskColor[r.risk_level] }"
          >
            <span class="lane-tag">车道 {{ (r.lane_id ?? 0) + 1 }}</span>
            <strong :style="{ color: riskColor[r.risk_level] }">{{ r.risk_level }}</strong>
            <span class="lane-score">评分 {{ r.risk_score }}</span>
          </article>
        </div>
      </div>

      <div v-if="environment" class="panel env-result">
        <header class="panel-head">
          <h3>环境识别结果（自动）</h3>
          <span class="muted">{{ environment.source === 'auto' ? '系统识别' : environment.source }}</span>
        </header>
        <EnvBars :env="environment" />
        <button type="button" class="accent small" @click="adoptEnvironment(environment)">
          采纳为人工标注并重跑
        </button>
      </div>

      <div class="data-grid">
        <div class="panel">
          <header class="panel-head">
            <h3>分车道分分钟交通指标</h3>
            <span class="muted">{{ metrics.length }} 条</span>
          </header>
          <div class="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>车道</th>
                  <th>分钟</th>
                  <th>流量</th>
                  <th>均速(km/h)</th>
                  <th>平均时距(s)</th>
                  <th>最小时距(s)</th>
                  <th>密度(辆/km)</th>
                  <th>货车占比</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="(m, i) in metrics" :key="i">
                  <td>{{ m.lane_id + 1 }}</td>
                  <td>{{ m.minute_bucket }}</td>
                  <td>{{ m.flow_count }}</td>
                  <td>{{ fmt(m.avg_speed_kmh) }}</td>
                  <td>{{ fmt(m.avg_headway_s, 2) }}</td>
                  <td>{{ fmt(m.min_headway_s, 2) }}</td>
                  <td>{{ fmt(m.density_per_km, 0) }}</td>
                  <td>{{ fmtPct(m.truck_ratio) }}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        <div class="panel">
          <header class="panel-head">
            <h3>车辆轨迹摘要</h3>
            <span class="muted">{{ tracks.length }} 辆</span>
          </header>
          <div class="type-chips">
            <span v-for="(count, type) in trackTypeCounts" :key="type" class="chip">
              {{ type }}: {{ count }}
            </span>
          </div>
          <div class="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>ID</th>
                  <th>车型</th>
                  <th>均速</th>
                  <th>峰值</th>
                  <th>车道</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="t in tracks.slice(0, 25)" :key="t.track_id">
                  <td>#{{ t.track_id }}</td>
                  <td>{{ t.vehicle_type ?? '—' }}</td>
                  <td>{{ fmt(t.avg_speed_kmh) }}</td>
                  <td>{{ fmt(t.max_speed_kmh) }}</td>
                  <td>{{ t.lane_id === null ? '—' : t.lane_id + 1 }}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </section>
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

.grid {
  display: grid;
  grid-template-columns: minmax(0, 1.5fr) minmax(260px, 0.9fr);
  gap: 1rem;
  align-items: start;
}

.feed,
.controls {
  border: 1px solid var(--line);
  background: var(--bg-panel);
  padding: 1rem;
}

.controls {
  display: grid;
  gap: 1rem;
}

.block h2 {
  margin: 0 0 0.35rem;
  font-family: 'Barlow Condensed', sans-serif;
  font-size: 1rem;
  letter-spacing: 0.07em;
  text-transform: uppercase;
}

.hint {
  margin: 0 0 0.75rem;
  color: var(--text-muted);
  font-size: 0.82rem;
}

.field {
  display: grid;
  gap: 0.35rem;
  margin-bottom: 0.75rem;
}

.field span {
  font-size: 0.82rem;
  color: var(--text-muted);
}

.field input,
.field select {
  padding: 0.55rem 0.65rem;
  border: 1px solid rgba(139, 148, 158, 0.45);
  background: var(--bg-base);
  color: var(--text);
  font-family: 'JetBrains Mono', monospace;
}

.stats {
  margin: 0;
  display: grid;
  gap: 0.55rem;
}

.stats div {
  display: flex;
  justify-content: space-between;
  gap: 1rem;
  padding-bottom: 0.45rem;
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

.actions {
  display: grid;
  gap: 0.65rem;
}

.primary,
.ghost {
  padding: 0.75rem 1rem;
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

.error {
  margin: 0;
  color: var(--danger);
  font-size: 0.875rem;
}

.detect-btn {
  margin-top: 0.25rem;
  align-self: start;
}

.accent {
  background: rgba(255, 107, 44, 0.12);
  border: 1px solid rgba(255, 107, 44, 0.45);
  color: var(--marking);
  font-family: 'Barlow Condensed', sans-serif;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  cursor: pointer;
}

.accent.small {
  padding: 0.4rem 0.7rem;
  font-size: 0.82rem;
  margin-top: 0.5rem;
}

.env-card {
  margin-top: 0.75rem;
  border: 1px solid rgba(255, 107, 44, 0.3);
  background: rgba(255, 107, 44, 0.06);
  padding: 0.75rem;
}

.env-eyebrow {
  margin: 0 0 0.4rem;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.7rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--text-muted);
}

.env-result {
  display: grid;
  gap: 0.5rem;
}

.results {
  margin-top: 1.5rem;
  display: grid;
  gap: 1rem;
}

.result-grid {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(0, 1.4fr);
  gap: 1rem;
  align-items: start;
}

.risk-card {
  border: 1px solid var(--line);
  background: var(--bg-panel);
  padding: 1.25rem;
}

.risk-eyebrow {
  margin: 0 0 0.35rem;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.72rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--text-muted);
}

.risk-level {
  margin: 0 0 0.4rem;
  font-family: 'Barlow Condensed', sans-serif;
  font-size: 3rem;
  line-height: 1;
}

.risk-score {
  margin: 0 0 0.75rem;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.85rem;
  color: var(--text-muted);
}

.factors {
  margin: 0;
  padding-left: 1.1rem;
  display: grid;
  gap: 0.3rem;
  font-size: 0.85rem;
  color: var(--text);
}

.lane-risks {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
  gap: 0.65rem;
}

.lane-risk {
  border: 1px solid var(--line);
  background: var(--bg-panel);
  padding: 0.75rem;
  display: grid;
  gap: 0.25rem;
}

.lane-tag {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.72rem;
  color: var(--text-muted);
}

.lane-risk strong {
  font-family: 'Barlow Condensed', sans-serif;
  font-size: 1.3rem;
}

.lane-score {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.78rem;
  color: var(--text-muted);
}

.data-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
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
  margin-bottom: 0.75rem;
}

.panel-head h3 {
  margin: 0;
  font-family: 'Barlow Condensed', sans-serif;
  font-size: 1.05rem;
  letter-spacing: 0.05em;
}

.muted {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.75rem;
  color: var(--text-muted);
}

.table-wrap {
  max-height: 320px;
  overflow: auto;
  border: 1px solid rgba(139, 148, 158, 0.2);
}

table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.8rem;
}

th,
td {
  padding: 0.4rem 0.55rem;
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

.type-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem;
  margin-bottom: 0.75rem;
}

.chip {
  padding: 0.25rem 0.55rem;
  border: 1px solid rgba(139, 148, 158, 0.35);
  border-radius: 999px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.74rem;
}

@media (max-width: 900px) {
  .grid,
  .result-grid,
  .data-grid {
    grid-template-columns: 1fr;
  }
}
</style>
