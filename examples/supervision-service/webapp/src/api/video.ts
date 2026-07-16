import { jobResultUrl, pollJobUntilComplete } from '@/api/jobs'

export interface HealthResponse {
  status: string
}

export interface TrackVideoOptions {
  file?: File
  uploadId?: string
  serverVideo?: string
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

export interface ServerVideo {
  name: string
  size: number
  previewUrl: string
  downloadUrl: string
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
 *
 * `onUploadProgress` reports the upload completion percentage (0-100) so the UI
 * can show a live progress bar. `fetch` does not expose upload progress events,
 * so this uses XMLHttpRequest under the hood.
 */
export async function uploadPreview(
  file: File,
  onUploadProgress?: (percent: number) => void,
): Promise<UploadPreviewResult> {
  return new Promise((resolve, reject) => {
    const formData = new FormData()
    formData.append('file', file)

    const xhr = new XMLHttpRequest()
    xhr.open('POST', '/api/v1/videos/preview')

    if (onUploadProgress) {
      xhr.upload.onprogress = (event) => {
        if (event.lengthComputable) {
          onUploadProgress(Math.round((event.loaded / event.total) * 100))
        }
      }
    }

    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          const data = JSON.parse(xhr.responseText) as {
            upload_id: string
            preview_url: string
          }
          resolve({ uploadId: data.upload_id, previewUrl: data.preview_url })
        } catch {
          reject(new Error('无法解析服务器响应。'))
        }
        return
      }

      let detail = `Upload failed: ${xhr.status}`
      try {
        const payload = JSON.parse(xhr.responseText) as { detail?: string }
        if (payload.detail) {
          detail = payload.detail
        }
      } catch {
        // ignore JSON parse errors
      }
      reject(new Error(detail))
    }

    xhr.onerror = () => reject(new Error('网络错误，上传失败。'))
    xhr.ontimeout = () => reject(new Error('上传超时。'))

    xhr.send(formData)
  })
}

export async function fetchServerVideos(): Promise<ServerVideo[]> {
  const response = await fetch('/api/v1/videos/server')
  if (!response.ok) {
    throw new Error(`Failed to load sample videos: ${response.status}`)
  }
  const data = (await response.json()) as Array<{
    name: string
    size: number
    preview_url: string
    download_url: string
  }>
  return data.map((item) => ({
    name: item.name,
    size: item.size,
    previewUrl: item.preview_url,
    downloadUrl: item.download_url,
  }))
}

export async function trackVideo({
  file,
  uploadId,
  serverVideo,
  confidenceThreshold = 0.3,
  iouThreshold = 0.7,
  onProgress,
  signal,
}: TrackVideoOptions): Promise<TrackVideoResult> {
  const formData = new FormData()
  if (serverVideo) {
    formData.append('server_video', serverVideo)
  } else if (uploadId) {
    formData.append('upload_id', uploadId)
  } else if (file) {
    formData.append('file', file)
  } else {
    throw new Error('Either file, uploadId, or serverVideo is required.')
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
