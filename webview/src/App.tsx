import { Canvas } from '@react-three/fiber'
import { useGLTF, OrbitControls, useAnimations } from '@react-three/drei'
import {
  Component,
  Suspense,
  useEffect,
  useRef,
  useState,
  type ChangeEvent,
  type DragEvent,
  type ErrorInfo,
  type ReactNode,
} from 'react'
import './global.css'
import './app.css'

const API_URL = (import.meta.env.VITE_API_URL ?? '').replace(/\/$/, '')
const ACCEPTED_EXTENSIONS = ['.gltf', '.glb', '.mp4']
const MAX_SIZE_MB = 200

/* ------------------------------------------------------------------ */
/* Model                                                               */
/* ------------------------------------------------------------------ */

function Model({ url }: { url: string }) {
  const gltf = useGLTF(url)
  const { actions, names } = useAnimations(gltf.animations, gltf.scene)

  useEffect(() => {
    if (names.length > 0) actions[names[0]]?.reset().play()
  }, [actions, names])

  useEffect(() => {
    return () => {
      gltf.scene.traverse((obj: any) => {
        obj.geometry?.dispose?.()
        const m = obj.material
        if (Array.isArray(m)) m.forEach((x: any) => x.dispose?.())
        else m?.dispose?.()
      })
    }
  }, [gltf])

  return <primitive object={gltf.scene} scale={[2, 2, 2]} />
}

/* ------------------------------------------------------------------ */
/* Error boundary                                                      */
/* ------------------------------------------------------------------ */

class CanvasErrorBoundary extends Component<
  { children: ReactNode; onReset: () => void },
  { error: Error | null }
> {
  state = { error: null as Error | null }
  static getDerivedStateFromError(error: Error) {
    return { error }
  }
  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error('Canvas error:', error, info)
  }
  handleRetry = () => {
    this.setState({ error: null })
    this.props.onReset()
  }
  render() {
    if (this.state.error) {
      return (
        <div className="canvas-overlay canvas-overlay--error">
          <p style={{ margin: 0 }}>Couldn't load this model.</p>
          <button onClick={this.handleRetry}>Try another file</button>
        </div>
      )
    }
    return this.props.children
  }
}

/* ------------------------------------------------------------------ */
/* Types + helpers                                                     */
/* ------------------------------------------------------------------ */

interface StageInfo {
  key: string
  label: string
  state: 'pending' | 'active' | 'done' | 'failed'
}

interface JobStatus {
  id: string
  filename: string
  status: 'queued' | 'running' | 'done' | 'error' | 'cancelled'
  stage: string
  message: string
  progress: number
  frames_total: number
  frames_processed: number
  frames_with_pose: number
  elapsed_seconds: number
  stages: StageInfo[]
  error: { code: string; message: string } | null
}

function formatSize(bytes: number) {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

function getExtension(name: string) {
  return name.split('.').pop()?.toLowerCase() ?? ''
}

/* ------------------------------------------------------------------ */
/* App                                                                 */
/* ------------------------------------------------------------------ */

export default function App() {
  const [file, setFile] = useState<File | null>(null)
  const [modelUrl, setModelUrl] = useState<string | null>(null)
  const [isUploading, setIsUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [job, setJob] = useState<JobStatus | null>(null)
  const [isDragging, setIsDragging] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const inputRef = useRef<HTMLInputElement>(null)
  const xhrRef = useRef<XMLHttpRequest | null>(null)
  const jobIdRef = useRef<string | null>(null)
  const pollRef = useRef<number | null>(null)

  const isViewing = modelUrl !== null
  const fileExt = file ? `.${getExtension(file.name)}` : ''
  const isVideo = fileExt === '.mp4'
  const isProcessing = isUploading || (job !== null && job.status === 'running')

  /* ---------- File selection ---------- */

  const validateFile = (f: File): string | null => {
    const ext = `.${getExtension(f.name)}`
    if (!ACCEPTED_EXTENSIONS.includes(ext)) {
      return `Unsupported file type. Accepted: ${ACCEPTED_EXTENSIONS.join(', ')}`
    }
    if (f.size > MAX_SIZE_MB * 1024 * 1024) {
      return `File is too large (max ${MAX_SIZE_MB} MB).`
    }
    return null
  }

  const selectFile = (f: File | undefined) => {
    if (!f) return
    const validationError = validateFile(f)
    if (validationError) {
      setError(validationError)
      setFile(null)
      return
    }
    setError(null)
    setFile(f)
  }

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) =>
    selectFile(e.target.files?.[0])

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setIsDragging(false)
    selectFile(e.dataTransfer.files?.[0])
  }

  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    if (!isDragging) setIsDragging(true)
  }

  const handleDragLeave = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setIsDragging(false)
  }

  /* ---------- Reset ---------- */

  const stopPolling = () => {
    if (pollRef.current !== null) {
      window.clearInterval(pollRef.current)
      pollRef.current = null
    }
  }

  const reset = () => {
    stopPolling()
    xhrRef.current?.abort()
    xhrRef.current = null

    const jobId = jobIdRef.current
    if (jobId && API_URL) {
      fetch(`${API_URL}/jobs/${jobId}`, { method: 'DELETE' }).catch(() => { })
    }
    jobIdRef.current = null

    if (modelUrl) URL.revokeObjectURL(modelUrl)
    setModelUrl(null)
    setFile(null)
    setError(null)
    setJob(null)
    setUploadProgress(0)
    setIsUploading(false)

    if (inputRef.current) inputRef.current.value = ''
  }

  /* ---------- Job polling ---------- */

  const startPolling = (jobId: string) => {
    stopPolling()
    pollRef.current = window.setInterval(async () => {
      try {
        const r = await fetch(`${API_URL}/jobs/${jobId}`)
        if (!r.ok) return
        const data: JobStatus = await r.json()
        setJob(data)

        if (data.status === 'done') {
          stopPolling()
          const glb = await fetch(`${API_URL}/jobs/${jobId}/result`)
          if (!glb.ok) {
            const body = await glb.json().catch(() => null)
            setError(body?.error?.message ?? 'Failed to download result.')
            return
          }
          const blob = await glb.blob()
          setModelUrl(URL.createObjectURL(blob))
        } else if (data.status === 'error') {
          stopPolling()
          setError(data.error?.message ?? 'Conversion failed.')
        } else if (data.status === 'cancelled') {
          stopPolling()
        }
      } catch {
        /* transient network error, keep polling */
      }
    }, 700)
  }

  /* ---------- Upload ---------- */

  const handleUpload = () => {
    if (!file) return
    setError(null)
    setJob(null)

    if (modelUrl) {
      URL.revokeObjectURL(modelUrl)
      setModelUrl(null)
    }

    const ext = getExtension(file.name)

    // Local formats — no round trip.
    if (ext === 'gltf' || ext === 'glb') {
      setModelUrl(URL.createObjectURL(file))
      return
    }

    if (ext === 'mp4') {
      if (!API_URL) {
        setError('API URL is not configured.')
        return
      }

      setIsUploading(true)
      setUploadProgress(0)

      const formData = new FormData()
      formData.append('uploaded_file', file)

      const xhr = new XMLHttpRequest()
      xhrRef.current = xhr
      xhr.responseType = 'json'

      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) {
          setUploadProgress(Math.round((e.loaded / e.total) * 100))
        }
      }

      xhr.onload = () => {
        setIsUploading(false)
        xhrRef.current = null
        if (xhr.status >= 200 && xhr.status < 300 && xhr.response?.job_id) {
          jobIdRef.current = xhr.response.job_id
          startPolling(xhr.response.job_id)
        } else {
          const msg =
            xhr.response?.error?.message ??
            `Upload failed (${xhr.status}).`
          setError(msg)
        }
      }

      xhr.onerror = () => {
        setIsUploading(false)
        xhrRef.current = null
        setError('Network error during upload.')
      }

      xhr.onabort = () => {
        setIsUploading(false)
        xhrRef.current = null
      }

      xhr.open('POST', `${API_URL}/jobs`)
      xhr.timeout = 10 * 60 * 1000
      xhr.send(formData)
    }
  }

  const handleCancel = async () => {
    // If upload is in flight, abort it. If job is running, DELETE it.
    if (xhrRef.current) {
      xhrRef.current.abort()
      xhrRef.current = null
      setIsUploading(false)
    }
    const jobId = jobIdRef.current
    if (jobId && API_URL) {
      try {
        await fetch(`${API_URL}/jobs/${jobId}`, { method: 'DELETE' })
      } catch {
        /* ignore */
      }
    }
    stopPolling()
    setJob(null)
    setUploadProgress(0)
    jobIdRef.current = null
    setError(null)
  }

  /* ---------- Keyboard shortcuts ---------- */

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'o') {
        e.preventDefault()
        if (!isViewing && !isProcessing) inputRef.current?.click()
      }
      if (e.key === 'Escape' && isViewing) reset()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isViewing, isProcessing])

  /* ------------------------------------------------------------------ */
  /* Render                                                              */
  /* ------------------------------------------------------------------ */

  // Determine what the progress panel should display.
  const showProgress = isProcessing
  const overallProgress = isUploading
    ? Math.round(uploadProgress * 0.4) // upload = first 40%
    : job
      ? Math.round(40 + job.progress * 0.6) // server work = remaining 60%
      : 0

  const progressLabel = isUploading
    ? `Uploading video… ${uploadProgress}%`
    : job?.message ?? 'Starting…'

  const uploadPanel = (
    <div className="upload-panel">
      <div
        className={[
          'dropzone',
          isDragging ? 'dropzone--active' : '',
          file ? 'dropzone--filled' : '',
        ]
          .filter(Boolean)
          .join(' ')}
        onClick={() => !file && !isProcessing && inputRef.current?.click()}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        role="button"
        tabIndex={0}
        onKeyDown={(e) =>
          !file &&
          !isProcessing &&
          (e.key === 'Enter' || e.key === ' ') &&
          inputRef.current?.click()
        }
      >
        <input
          ref={inputRef}
          type="file"
          accept=".gltf,.glb,.mp4"
          onChange={handleFileChange}
          hidden
          aria-label="Choose a 3D file"
        />

        {file ? (
          <div className="file-info">
            <div className="file-icon">{isVideo ? '🎬' : '📦'}</div>
            <div className="file-meta">
              <span className="file-name" title={file.name}>
                {file.name}
              </span>
              <span className="file-size">{formatSize(file.size)}</span>
            </div>
            <button
              type="button"
              className="icon-button"
              onClick={(e) => {
                e.stopPropagation()
                reset()
              }}
              aria-label="Remove file"
              disabled={isProcessing}
            >
              ✕
            </button>
          </div>
        ) : (
          <div className="dropzone-empty">
            <div className="dropzone-icon">⬆</div>
            <p className="dropzone-title">Drop your file here</p>
            <p className="dropzone-hint">
              or click to browse — .glb, .gltf, .mp4 (max {MAX_SIZE_MB} MB)
            </p>
          </div>
        )}
      </div>

      {error && (
        <div className="error-banner" role="alert">
          <span>{error}</span>
          <button type="button" onClick={() => setError(null)}>
            Dismiss
          </button>
        </div>
      )}

      {showProgress && (
        <div className="job-panel" aria-live="polite">
          <div className="progress">
            <div className="progress-row">
              <span>{progressLabel}</span>
              <span>{overallProgress}%</span>
            </div>
            <div className="progress-track">
              <div
                className="progress-fill"
                style={{ width: `${overallProgress}%` }}
              />
            </div>
          </div>

          {job && job.frames_total > 0 && (
            <div className="job-stats">
              <span>
                <strong>{job.frames_processed}</strong> / {job.frames_total}{' '}
                frames
              </span>
              <span>
                <strong>{job.frames_with_pose}</strong> poses detected
              </span>
              <span>{job.elapsed_seconds.toFixed(1)}s elapsed</span>
            </div>
          )}

          {job && (
            <ul className="stage-list">
              {job.stages.map((s) => (
                <li key={s.key} className={`stage stage--${s.state}`}>
                  <span className="stage-dot" />
                  <span className="stage-label">{s.label}</span>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}

      <div style={{ display: 'flex', gap: 10 }}>
        <button
          type="button"
          className="primary-button"
          style={{ flex: 1 }}
          disabled={!file || isProcessing}
          onClick={handleUpload}
        >
          {isProcessing ? (
            <>
              <span className="spinner spinner--sm" /> Processing…
            </>
          ) : isVideo ? (
            'Convert & Preview'
          ) : (
            'Preview Model'
          )}
        </button>

        {isProcessing && (
          <button type="button" className="ghost-button" onClick={handleCancel}>
            Cancel
          </button>
        )}
      </div>
    </div>
  )

  return (
    <div
      className={`app space-mono-regular ${isViewing ? 'app--viewing' : ''}`}
    >
      {!isViewing && (
        <header className="header">
          <h1 className="titles">3D Viewer</h1>
          <p className="subtitle">
            Upload a <code>.mp4</code> video and we'll convert it to an
            interactive 3D model. Or bring your own <code>.glb</code> /{' '}
            <code>.gltf</code>.
          </p>
        </header>
      )}

      {!isViewing ? (
        uploadPanel
      ) : (
        <div className="viewer">
          <div className="viewer-toolbar">
            <span className="viewer-filename" title={file?.name}>
              {file?.name}
              {job && job.frames_with_pose > 0 && (
                <>
                  {' · '}
                  <span style={{ color: 'var(--text-faint)' }}>
                    {job.frames_with_pose} of {job.frames_total} frames tracked
                  </span>
                </>
              )}
            </span>
            <button type="button" className="ghost-button" onClick={reset}>
              Upload another
            </button>
          </div>

          <div className="canvas-wrapper">
            <CanvasErrorBoundary onReset={reset}>
              <Suspense
                fallback={
                  <div className="canvas-overlay">
                    <span className="spinner" />
                    <span>Loading model…</span>
                  </div>
                }
              >
                <Canvas camera={{ position: [0, 1, 5], fov: 45 }}>
                  <ambientLight intensity={0.7} />
                  <directionalLight position={[5, 5, 5]} intensity={1.1} />
                  <pointLight position={[0, 2, 2]} intensity={8} />
                  <Model url={modelUrl!} />
                  <OrbitControls enablePan makeDefault />
                </Canvas>
              </Suspense>
            </CanvasErrorBoundary>
          </div>

          <p className="canvas-hint">
            Drag to rotate · Scroll to zoom · Right-drag to pan
          </p>
        </div>
      )}
    </div>
  )
}