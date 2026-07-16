import { jobResultUrl, pollJobUntilComplete } from '@/api/jobs'

export interface HealthResponse {
  status: string
}

export interface TrackVideoOptions {
  file?: File
  uploadId?: string
  confidenceThreshold?: number
  iouThreshold?: number
  onProgress?: (progress: number, currentFrame: number, totalFrames: number) => void
  signal?: AbortSignal
}

export interface TrackVideoResult {
  jobId: string
  resultUrl: string
}

export interface UploadPreviewResult {
  uploadId: string
  previewUrl: string
}

export async function fetchHealth(): Promise<HealthResponse> {
  const response = await fetch('/health')
  if (!response.ok) {
    throw new Error(`Health check failed: ${response.status}`)
  }
  return response.json() as Promise<HealthResponse>
}

/**
 * Upload a video and get back a browser-playable preview URL. Browsers cannot
 * decode formats like .avi natively, so the server transcodes the source to an
 * H.264 MP4. The returned uploadId can be reused to start processing without
 * uploading the file a second time.
 */
export async function uploadPreview(file: File): Promise<UploadPreviewResult> {
  const formData = new FormData()
  formData.append('file', file)

  const response = await fetch('/api/v1/videos/preview', {
    method: 'POST',
    body: formData,
  })

  if (!response.ok) {
    let detail = `Upload failed: ${response.status}`
    try {
      const payload = (await response.json()) as { detail?: string }
      if (payload.detail) {
        detail = payload.detail
      }
    } catch {
      // ignore JSON parse errors
    }
    throw new Error(detail)
  }

  const data = (await response.json()) as { upload_id: string; preview_url: string }
  return { uploadId: data.upload_id, previewUrl: data.preview_url }
}

export async function trackVideo({
  file,
  uploadId,
  confidenceThreshold = 0.3,
  iouThreshold = 0.7,
  onProgress,
  signal,
}: TrackVideoOptions): Promise<TrackVideoResult> {
  const formData = new FormData()
  if (uploadId) {
    formData.append('upload_id', uploadId)
  } else if (file) {
    formData.append('file', file)
  } else {
    throw new Error('Either file or uploadId is required.')
  }
  formData.append('confidence_threshold', String(confidenceThreshold))
  formData.append('iou_threshold', String(iouThreshold))

  const response = await fetch('/api/v1/videos/track', {
    method: 'POST',
    body: formData,
    signal,
  })

  if (!response.ok) {
    let detail = `Request failed: ${response.status}`
    try {
      const payload = (await response.json()) as { detail?: string }
      if (payload.detail) {
        detail = payload.detail
      }
    } catch {
      // ignore JSON parse errors
    }
    throw new Error(detail)
  }

  const created = (await response.json()) as { job_id: string }

  await pollJobUntilComplete(
    created.job_id,
    (job) => {
      onProgress?.(job.progress, job.current_frame, job.total_frames)
    },
    signal,
  )

  return {
    jobId: created.job_id,
    resultUrl: jobResultUrl(created.job_id),
  }
}
