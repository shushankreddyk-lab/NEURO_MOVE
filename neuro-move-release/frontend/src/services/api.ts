import axios from 'axios'

const API_BASE = (import.meta as any).env?.VITE_API_URL || 'http://localhost:8000'

const api = axios.create({
  baseURL: API_BASE,
  headers: { 'Content-Type': 'application/json' },
})

api.interceptors.request.use((config) => {
  const turnstileToken = (window as any).turnstileToken
  if (turnstileToken) config.headers['X-Turnstile-Token'] = turnstileToken
  return config
})

export interface ModelInfo {
  name: string
  description: string
  classes: string[]
  input_shape: number[]
}

export interface InferenceRequest {
  model_name: string
  eeg_data: number[][][]
  sampling_rate?: number
  channels?: string[]
  turnstile_token?: string
}

export interface InferenceResponse {
  predictions: number[]
  probabilities: number[][]
  class_labels: string[]
  latency_ms: number
  model_used: string
}

export interface ModelLoadResponse {
  success: boolean
  message: string
  model_info?: ModelInfo
}

export interface PreprocessRequest {
  eeg_data: number[][][]
  sampling_rate?: number
  target_rate?: number
  low_freq?: number
  high_freq?: number
  turnstile_token?: string
}

export interface PreprocessResponse {
  processed_data: number[][][]
  sampling_rate: number
  channels: number
  samples: number
}

export const apiService = {
  async getHealth() {
    const res = await api.get('/health')
    return res.data
  },

  async getDatasets() {
    const res = await api.get('/datasets')
    return res.data
  },

  async getModels(): Promise<ModelInfo[]> {
    // Demo backend has no /models registry — return static catalogue wired to real engines
    return [
      { name: 'MiniRocket', description: 'GPU MiniRocket + MLP head (10k kernels)', classes: ['Left Fist', 'Right Fist', 'Both Fists', 'Both Feet'], input_shape: [22, 656] },
      { name: 'EEGNet', description: 'Compact depthwise-separable CNN', classes: ['Left Fist', 'Right Fist', 'Both Fists', 'Both Feet'], input_shape: [22, 656] },
      { name: 'CNN-LSTM', description: 'Spatio-temporal CNN + LSTM', classes: ['Left Fist', 'Right Fist', 'Both Fists', 'Both Feet'], input_shape: [22, 656] },
      { name: 'Conformer', description: 'CNN + Transformer attention', classes: ['Left Fist', 'Right Fist', 'Both Fists', 'Both Feet'], input_shape: [22, 656] },
    ]
  },

  async loadModel(modelName: string) {
    const res = await api.post('/models/load', { model_name: modelName })
    return res.data
  },

  async inference(request: InferenceRequest): Promise<InferenceResponse> {
    const res = await api.post('/inference', request)
    return res.data
  },

  async preprocess(request: {
    eeg_data: number[][][]
    sampling_rate?: number
    target_rate?: number
    low_freq?: number
    high_freq?: number
  }) {
    const res = await api.post('/preprocess', request)
    return res.data
  },

  // ---- Demo-backend helpers (JSX pages: Dashboard/Datasets/Training/Inference) ----
  async submitTraining(body: { dataset: string; engine: string; epochs?: number; batch_size?: number; learning_rate?: number }) {
    const res = await api.post('/train', body)
    return { data: res.data }
  },

  async getJobStatus(jobId: string) {
    const res = await api.get(`/train/${jobId}`)
    return { data: res.data }
  },

  async runPreview(jobId: string, body: { dataset: string; engine: string; epochs?: number; batch_size?: number; learning_rate?: number }) {
    const res = await api.post(`/train/${jobId}/run-preview`, body)
    return { data: res.data }
  },

  async infer(body: { engine: string; dataset: string; sample_count?: number; channels?: number; samples?: number }) {
    const res = await api.post('/infer', body)
    return { data: res.data }
  },

  async evaluate(body: { dataset: string; engine: string }) {
    const res = await api.post('/evaluate', body)
    return { data: res.data }
  },
}

export default apiService

// ---- Named exports for legacy JSX pages (Dashboard/Datasets/Training/Inference) ----
export async function getHealth() {
  const data = await apiService.getHealth()
  return { data }
}

export async function getDatasets() {
  const data = await apiService.getDatasets()
  return { data }
}

export async function submitTraining(body: { dataset: string; engine: string; epochs?: number; batch_size?: number; learning_rate?: number }) {
  return apiService.submitTraining(body)
}

export async function getJobStatus(jobId: string) {
  return apiService.getJobStatus(jobId)
}

export async function runPreview(jobId: string, body: { dataset: string; engine: string; epochs?: number; batch_size?: number; learning_rate?: number }) {
  return apiService.runPreview(jobId, body)
}

export async function infer(body: { engine: string; dataset: string; sample_count?: number; channels?: number; samples?: number }) {
  return apiService.infer(body)
}

export async function evaluate(body: { dataset: string; engine: string }) {
  return apiService.evaluate(body)
}