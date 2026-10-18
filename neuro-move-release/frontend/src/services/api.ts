import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  headers: { 'Content-Type': 'application/json' },
})

api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token')
  if (token) config.headers.Authorization = `Bearer ${token}`
  return config
})

api.interceptors.response.use(
  (res) => res,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('token')
      window.location.href = '/login'
    }
    return Promise.reject(error)
  }
)

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
}

export interface PreprocessResponse {
  processed_data: number[][][]
  sampling_rate: number
  channels: number
  samples: number
}

export const apiService = {
  async login(username: string, password: string) {
    const res = await axios.post('/api/auth/login', { username, password })
    return res.data
  },

  async getModels(): Promise<ModelInfo[]> {
    const res = await axios.get('/models')
    return res.data
  },

  async loadModel(modelName: string) {
    const res = await axios.post('/models/load', { model_name: modelName })
    return res.data
  },

  async inference(request: InferenceRequest): Promise<InferenceResponse> {
    const res = await axios.post('/inference', request)
    return res.data
  },

  async preprocess(request: {
    eeg_data: number[][][]
    sampling_rate?: number
    target_rate?: number
    low_freq?: number
    high_freq?: number
  }): Promise<{
    processed_data: number[][][]
    sampling_rate: number
    channels: number
    samples: number
  }> {
    const res = await axios.post('/preprocess', request)
    return res.data
  },
}

export const authService = {
  async login(username: string, password: string) {
    const res = await axios.post('/api/auth/login', { username, password })
    return res.data
  },

  logout() {
    localStorage.removeItem('token')
  },

  isAuthenticated(): boolean {
    return !!localStorage.getItem('token')
  },

  getToken(): string | null {
    return localStorage.getItem('token')
  },
}

export default apiService