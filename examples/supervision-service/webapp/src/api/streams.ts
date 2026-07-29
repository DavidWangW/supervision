import type { SourcePoint } from '@/api/speed'
import type { EnvProb } from '@/api/environment'

export type { SourcePoint } from '@/api/speed'

export interface StreamCreateParams {
  name: string
  url: string
  sourcePoints: SourcePoint[]
  targetWidth: number
  targetHeight: number
  laneCount: number
  weather?: string
  roadCondition?: string
  visibility?: string
  sampleFps: number
  confidenceThreshold: number
  iouThreshold: number
}

export interface StreamRecord {
  id: string
  name: string
  url: string
  source_points: number[][]
  target_width: number
  target_height: number
  lane_count: number
  weather: string | null
  road_condition: string | null
  visibility: string | null
  sample_fps: number
  confidence_threshold: number
  iou_threshold: number
  status: string
  error_message: string | null
  running: boolean
  runtime?: {
    fps_actual: number
    frames_processed: number
    started_at: string
  }
}

export interface PreviewFrame {
  width: number
  height: number
  image: string
}

export interface LaneLiveMetric {
  lane_id: number
  vehicle_count: number
  flow_total: number
  avg_speed_kmh: number | null
  min_headway_s: number | null
  density: number | null
  truck_ratio: number | null
}

export interface LiveRisk {
  overall: {
    risk_level: string
    risk_score: number
    probability: number | null
    factors: string[]
  } | null
  lanes: { lane_id: number; risk_level: string; risk_score: number }[]
}

export interface LiveSnapshot {
  frame_index: number
  t_sec: number
  minute_bucket: number
  total_vehicles: number
  lanes: LaneLiveMetric[]
  risk: LiveRisk | null
  environment?: EnvProb | null
  weather?: string | null
  road_condition?: string | null
  vlm_active?: boolean | null
}

export async function fetchPreviewFrame(url: string): Promise<PreviewFrame> {
  const response = await fetch('/api/v1/streams/preview-frame', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ url }),
  })
  if (!response.ok) {
    const payload = (await response.json().catch(() => ({}))) as { detail?: string }
    throw new Error(payload.detail ?? `获取预览帧失败: ${response.status}`)
  }
  return response.json() as Promise<PreviewFrame>
}

export async function createStream(params: StreamCreateParams): Promise<StreamRecord> {
  const response = await fetch('/api/v1/streams', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      name: params.name,
      url: params.url,
      source_points: params.sourcePoints,
      target_width: params.targetWidth,
      target_height: params.targetHeight,
      lane_count: params.laneCount,
      weather: params.weather || null,
      road_condition: params.roadCondition || null,
      visibility: params.visibility || null,
      sample_fps: params.sampleFps,
      confidence_threshold: params.confidenceThreshold,
      iou_threshold: params.iouThreshold,
    }),
  })
  if (!response.ok) {
    const payload = (await response.json().catch(() => ({}))) as { detail?: string }
    throw new Error(payload.detail ?? `创建视频流失败: ${response.status}`)
  }
  return response.json() as Promise<StreamRecord>
}

export async function listStreams(): Promise<StreamRecord[]> {
  const response = await fetch('/api/v1/streams')
  if (!response.ok) {
    throw new Error(`加载视频流列表失败: ${response.status}`)
  }
  const data = (await response.json()) as { items: StreamRecord[] }
  return data.items
}

export async function startStream(id: string): Promise<StreamRecord> {
  const response = await fetch(`/api/v1/streams/${id}/start`, { method: 'POST' })
  if (!response.ok) throw new Error(`启动视频流失败: ${response.status}`)
  return response.json() as Promise<StreamRecord>
}

export async function stopStream(id: string): Promise<StreamRecord> {
  const response = await fetch(`/api/v1/streams/${id}/stop`, { method: 'POST' })
  if (!response.ok) throw new Error(`停止视频流失败: ${response.status}`)
  return response.json() as Promise<StreamRecord>
}

export async function deleteStream(id: string): Promise<void> {
  const response = await fetch(`/api/v1/streams/${id}`, { method: 'DELETE' })
  if (!response.ok && response.status !== 204) {
    throw new Error(`删除视频流失败: ${response.status}`)
  }
}

export function streamMjpegUrl(id: string): string {
  return `/api/v1/streams/${id}/mjpeg`
}

export function streamWsUrl(id: string): string {
  const proto = window.location.protocol === 'https:' ? 'wss' : 'ws'
  return `${proto}://${window.location.host}/api/v1/streams/${id}/ws`
}
