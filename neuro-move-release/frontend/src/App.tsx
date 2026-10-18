import React, { useState, useEffect } from 'react'
import axios from 'axios'
import Dashboard from './components/Dashboard'
import './App.css'

const api = axios.create({
  baseURL: '/api',
  headers: { 'Content-Type': 'application/json' },
)

// Add Turnstile token to requests if available
api.interceptors.request.use((config) => {
  const token = (window as any).turnstileToken
  if (token) config.headers['X-Turnstile-Token'] = token
  return config
})

function App() {
  return (
    <div style={styles.app}>
      <Dashboard />
    </div>
  )
}

export default App

const styles: Record<string, React.CSSProperties> = {
  app: {
    minHeight: '100vh',
    display: 'flex',
    flexDirection: 'column',
  },
}