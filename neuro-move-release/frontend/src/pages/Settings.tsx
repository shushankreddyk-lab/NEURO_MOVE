import React, { useState } from 'react'

export function Settings() {
  const [theme, setTheme] = useState('light')
  const [notifications, setNotifications] = useState(true)
  const [emailAlerts, setEmailAlerts] = useState(false)

  return (
    <div className="main-content">
      <div className="page-header">
        <h1 className="page-title">Settings</h1>
        <p className="page-subtitle">Configure your NeuroMove platform preferences</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-1.5rem">
        <div className="lg:col-span-2">
          <div className="card">
            <div className="card-header">
              <h3 className="card-title">General Settings</h3>
            </div>
            <div className="card-body space-y-4">
              <div className="flex items-center justify-between p-3 rounded-lg bg-slate-50 border border-slate-200">
                <div>
                  <p className="text-sm font-medium text-slate-700">Application Theme</p>
                  <p className="text-xs text-slate-500">Light or dark interface</p>
                </div>
                <div className="flex gap-2">
                  <button className={`px-3 py-1.5 rounded-md text-sm font-medium border border-slate-300 ${theme === 'light' ? 'bg-white border-sky-500 text-sky-600 shadow-sm' : 'bg-white text-slate-600'}`} onClick={() => setTheme('light')}>Light</button>
                  <button className="px-3 py-1.5 rounded-md text-sm font-medium border border-transparent ${theme === 'dark' ? 'bg-sky-100 text-sky-700' : 'bg-white text-slate-600'}" onClick={() => setTheme('dark')}>Dark</button>
                </div>
              </div>
              <div className="flex items-center justify-between p-3 rounded-lg bg-slate-50 border border-slate-200">
                <div>
                  <p className="text-sm font-medium text-slate-700">Email Notifications</p>
                  <p className="text-xs text-slate-500">Receive email alerts for training completion</p>
                </div>
                <div className="flex items-center">
                  <label className="toggle-switch">
                    <input type="checkbox" checked={emailAlerts} onChange={(e) => setEmailAlerts(e.target.checked)} />
                    <span className="toggle-slider"></span>
                  </label>
                </div>
              </div>
              <div className="flex items-center justify-between p-3 rounded-lg bg-slate-50 border border-slate-200">
                <div>
                  <p className="text-sm font-medium text-slate-700">Desktop Notifications</p>
                  <p className="text-xs text-slate-500">Show browser notifications</p>
                </div>
                <div className="flex items-center">
                  <label className="toggle-switch">
                    <input type="checkbox" checked={notifications} onChange={(e) => setNotifications(e.target.checked)} />
                    <span className="toggle-slider"></span>
                  </label>
                </div>
              </div>
              <div className="flex items-center justify-between p-3 rounded-lg bg-slate-50 border border-slate-200">
                <div>
                  <p className="text-sm font-medium text-slate-700">Auto-save Models</p>
                  <p className="text-xs text-slate-500">Automatically save model checkpoints</p>
                </div>
                <div className="flex items-center">
                  <label className="toggle-switch">
                    <input type="checkbox" checked onChange={() => {}} />
                    <span className="toggle-slider"></span>
                  </label>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="lg:col-span-1">
          <div className="card">
            <div className="card-header">
              <h3 className="card-title">Account</h3>
            </div>
            <div className="card-body space-y-4">
              <div className="p-3 rounded-lg bg-slate-50 border border-slate-200 flex items-center gap-3">
                <div className="w-10 h-10 rounded-full bg-sky-100 flex items-center justify-center text-sky-600 text-xl font-bold">A</div>
                <div>
                  <p className="text-sm font-medium text-slate-900">Admin User</p>
                  <p className="text-xs text-slate-500">admin@neuromove.ai</p>
                </div>
              </div>
              <div className="grid grid-cols-2 gap-3">
                <button className="btn btn-outline btn-sm w-full text-left">
                  <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth="2"><path d="M16 7a4 4 0 11-8 0 4 4 0 014 4zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"/></svg>
                  Change Password
                </button>
                <button className="btn btn-outline btn-sm w-full text-left">
                  <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth="2"><path d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v6a2 2 0 01-2 2h-2a2 2 0 01-2-2zM12 15a3 3 0 11-6 0 3 3 0 016 0z"/></svg>
                  Change Email
                </button>
                <button className="btn btn-outline btn-sm w-full text-left">
                  <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth="2"><path d="M12 1a2 2 0 012 2v4a2 2 0 01-2 2H2a2 2 0 01-2-2V3a2 2 0 012-2zM19 14a2 2 0 01-2-2h-1v4a2 2 0 002 2h2a2 2 0 002-2v-4a2 2 0 00-2-2z"/></svg>
                  Account Security
                </button>
                <button className="btn btn-outline btn-sm w-full text-left">
                  <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth="2"><path d="M3 5a2 2 0 012-2h3.5a2 2 0 012 2v12h-3.5a2 2 0 01-2-2zM20 21a2 2 0 01-2-2v-3.5a2 2 0 00-2-2H5a2 2 0 00-2 2V21a2 2 0 002 2h12a2 2 0 002-2z"/></svg>
                  Integrations
                </button>
              </div>
              <div className="pt-3 border-t border-slate-200">
                <button className="btn btn-outline btn-sm w-full text-left text-slate-600">
                  Export Settings
                </button>
                <button className="btn btn-danger btn-sm w-full mt-2">
                  Reset to Defaults
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}