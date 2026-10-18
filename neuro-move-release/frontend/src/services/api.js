import axios from 'axios'

const API_BASE_URL = 'http://localhost:8000'

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

export const getHealth = () => api.get('/health')
export const getDatasets = () => api.get('/datasets')
export const preprocess = (data) => api.post('/preprocess', data)
export const submitTraining = (data) => api.post('/train', data)
export const getJobStatus = (jobId) => api.get(`/train/${jobId}`)
export const runPreview = (jobId, data) => api.post(`/train/${jobId}/run-preview`, data)
export const infer = (data) => api.post('/infer', data)
export const evaluate = (data) => api.post('/evaluate', data)

export default api
