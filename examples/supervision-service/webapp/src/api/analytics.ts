import type { SourcePoint } from '@/api/speed'
import type { ProcessingJobRecord } from '@/api/records'

export interface AnalyzeParams {
  uploadId?: string
  file?: File
  serverVideo?: string
  sourcePoints: SourcePoint[]
  targetWidth: number
  targetHeight: number
  laneCount: number
  weather?: string
  roadCondition?: string
  visibility?: string
  confidenceThreshold: number
  iouThreshold: number
}

export interface JobCreatedResponse {
  job_id: string
  upload_id: string
  status: string
}

export interface TrafficMetric {
  lane_id: number
  minute_bucket: number
  flow_count: number
  avg_speed_kmh: number | null
  avg_headway_s: number | null
  min_headway_s: number | null
  density_per_km: number | null
  truck_ratio: number | null
  vehicle_count: number
}

export interface VehicleTrack {
  track_id: number
  class_name: string | null
  vehicle_type: string | null
  first_seen: number
  last_seen: number
  avg_speed_kmh: number | null
  max_speed_kmh: number | null
  lane_id: number | null
}

export interface RiskAssessment {
  id: number
  lane_id: number | null
  risk_level: string
  risk_score: number
  probability: number | null
  factors: string[]
  weather: string | null
  road_condition: string | null
  assessed_at: string
}

export async function analyzeTraffic(params: AnalyzeParams): Promise<JobCreatedResponse> {
  const form = new FormData()
  form.append('source_points', JSON.stringify(params.sourcePoints))
  form.append('target_width', String(params.targetWidth))
  form.append('target_height', String(params.targetHeight))
  form.append('lane_count', String(params.laneCount))
  if (params.weather) form.append('weather', params.weather)
  if (params.roadCondition) form.append('road_condition', params.roadCondition)
  if (params.visibility) form.append('visibility', params.visibility)
  form.append('confidence_threshold', String(params.confidenceThreshold))
  form.append('iou_threshold', String(params.iouThreshold))
  if (params.uploadId) form.append('upload_id', params.uploadId)
  if (params.serverVideo) form.append('server_video', params.serverVideo)
  if (params.file) form.append('file', params.file)

  const response = await fetch('/api/v1/videos/analyze', {
    method: 'POST',
    body: form,
  })
  if (!response.ok) {
    const detail = await response.text()
    throw new Error(`综合分析提交失败: ${response.status} ${detail}`)
  }
  return response.json() as Promise<JobCreatedResponse>
}

export async function fetchJob(jobId: string): Promise<ProcessingJobRecord> {
  const response = await fetch(`/api/v1/records/jobs/${jobId}`)
  if (!response.ok) {
    throw new Error(`无法获取任务状态: ${response.status}`)
  }
  return response.json() as Promise<ProcessingJobRecord>
}

export async function fetchTrafficMetrics(uploadId: string): Promise<TrafficMetric[]> {
  const response = await fetch(
    `/api/v1/analytics/traffic-metrics?upload_id=${encodeURIComponent(uploadId)}`,
  )
  if (!response.ok) {
    throw new Error(`加载交通指标失败: ${response.status}`)
  }
  const data = (await response.json()) as { metrics: TrafficMetric[] }
  return data.metrics
}

export async function fetchVehicleTracks(uploadId: string): Promise<VehicleTrack[]> {
  const response = await fetch(
    `/api/v1/analytics/vehicle-tracks?upload_id=${encodeURIComponent(uploadId)}`,
  )
  if (!response.ok) {
    throw new Error(`加载车辆轨迹失败: ${response.status}`)
  }
  const data = (await response.json()) as { tracks: VehicleTrack[] }
  return data.tracks
}

export async function fetchRiskAssessments(uploadId: string): Promise<RiskAssessment[]> {
  const response = await fetch(
    `/api/v1/analytics/risk?upload_id=${encodeURIComponent(uploadId)}`,
  )
  if (!response.ok) {
    throw new Error(`加载风险评估失败: ${response.status}`)
  }
  const data = (await response.json()) as { risk: RiskAssessment[] }
  return data.risk
}
