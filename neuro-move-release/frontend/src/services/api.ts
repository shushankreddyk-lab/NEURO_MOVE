import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
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
  async getModels(): Promise<ModelInfo[]> {
    const res = await api.get('/models')
    return res.data
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
}

export default apiService