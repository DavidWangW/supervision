export interface EnvProb {
  weather: string
  weather_probs: Record<string, number>
  road_condition: string
  road_probs: Record<string, number>
  visibility: string
  visibility_probs: Record<string, number>
  is_night: boolean
  traffic_condition?: string
  description?: string
  // Whether the frame clearly shows a traffic accident. Only true when the
  // evidence is unambiguous; uncertain frames stay false (and accident_desc empty).
  has_accident?: boolean | null
  accident_desc?: string | null
  model: string
  features: {
    brightness: number
    contrast: number
    fog_index: number
    rain_index: number
    snow_index: number
    specular_index: number
    is_night: boolean
  }
}

export interface EnvironmentReading extends EnvProb {
  id?: number
  observed_at?: string
  source?: string
  details?: EnvProb
}

export interface EnvironmentLabels {
  weather: string[]
  road: string[]
  visibility: string[]
  model: string
}

export async function fetchLatestEnvironment(uploadId: string): Promise<EnvironmentReading | null> {
  const response = await fetch(
    `/api/v1/analytics/environment/latest?upload_id=${encodeURIComponent(uploadId)}`,
  )
  if (!response.ok) {
    throw new Error(`加载环境识别失败: ${response.status}`)
  }
  const data = (await response.json()) as { reading: EnvironmentReading | null }
  return data.reading
}

export async function detectEnvironment(image: string): Promise<EnvProb> {
  const response = await fetch('/api/v1/environment/detect', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image }),
  })
  if (!response.ok) {
    const payload = (await response.json().catch(() => ({}))) as { detail?: string }
    throw new Error(payload.detail ?? `环境识别失败: ${response.status}`)
  }
  return response.json() as Promise<EnvProb>
}

export async function fetchEnvironmentLabels(): Promise<EnvironmentLabels> {
  const response = await fetch('/api/v1/environment/labels')
  if (!response.ok) {
    throw new Error(`加载环境标签失败: ${response.status}`)
  }
  return response.json() as Promise<EnvironmentLabels>
}
