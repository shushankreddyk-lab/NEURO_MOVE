import { Children, ReactNode } from 'react'
import Link from './NavItem'

export interface AppLayoutProps {
  children: ReactNode
  sidebar?: boolean
}

export const AppLayout: React.FC<AppLayoutProps> = ({ children, sidebar = true }) => {
  return (
    <div className="flex h-screen bg-slate-50">
      {sidebar && <Sidebar />}
      <div className="flex flex-col flex-1 h-full overflow-hidden">
        <Header />
        <main className="flex-1 h-full overflow-y-auto bg-slate-50 p-6">
          <div className="max-w-7xl mx-auto">
            {children}
          </div>
        </main>
      </div>
    </div>
  )
}

interface SidebarProps {
  items?: NavItem[]
  activeItem?: string
}

interface NavItem {
  label: string
  icon?: ReactNode
  path: string
  active?: boolean
  onClick?: () => void
}

export const Sidebar: React.FC<SidebarProps> = ({ items, activeItem }) => {
  const defaultItems: NavItem[] = [
    { label: 'Dashboard', path: '/', icon: <DashboardIcon />, active: true },
    { label: 'Datasets', path: '/datasets', icon: <DatasetIcon /> },
    { label: 'Training', path: '/training', icon: <TrainingIcon /> },
    { label: 'Inference', path: '/inference', icon: <InferenceIcon /> },
    { label: 'Models', path: '/models', icon: <ModelsIcon /> },
    { label: 'Analytics', path: '/analytics', icon: <AnalyticsIcon /> },
    { label: 'Settings', path: '/settings', icon: <SettingsIcon /> },
  ]

  const navItems = items || defaultItems

  return (
    <aside className="w-64 bg-white border-r border-slate-200 flex flex-col h-full">
      <div className="p-4 border-b border-slate-200">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 bg-sky-500 rounded-lg flex items-center justify-center">
            <svg className="w-5 h-5 text-white" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M12 2L2 7l10 5 10-5-10-5z" />
              <path d="M2 17l10 5 10-5" />
              <path d="M2 12l10 5 10-5" />
            </svg>
          </div>
          <span className="text-lg font-bold text-slate-900 tracking-tight">NeuroMove</span>
        </div>
        <p className="text-xs text-slate-500 mt-1">Enterprise MI-BCI Platform</p>
      </div>
      <nav className="flex-1 p-3 space-y-1">
        {navItems.map((item) => (
          <NavItem
            key={item.path}
            label={item.label}
            icon={item.icon}
            path={item.path}
            active={item.active || (activeItem === item.path)}
            onClick={item.onClick}
          />
        ))}
      </nav>
      <div className="p-3 border-t border-slate-200">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-full bg-slate-100 flex items-center justify-center text-slate-600 text-xs font-medium">AD</div>
          <div>
            <div className="text-sm font-medium text-slate-900">Admin User</div>
            <div className="text-xs text-slate-500">admin@neuromove.ai</div>
          </div>
        </div>
      </div>
    </aside>
  )
}

function DashboardIcon() {
  return (
    <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <rect x="3" y="3" width="7" height="7" rx="1" />
      <rect x="14" y="3" width="7" height="7" rx="1" />
      <rect x="3" y="14" width="7" height="7" rx="1" />
      <rect x="14" y="14" width="7" height="7" rx="1" />
    </svg>
  )
}

function DatasetIcon() {
  return (
    <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <line x1="22" y1="2" x2="22" y2="22" />
      <path d="M22 2l-7 7-7-7" />
      <ellipse cx="5" cy="12" rx="2" ry="3" />
      <ellipse cx="12" cy="5" rx="3" ry="2" />
      <ellipse cx="12" cy="19" rx="3" ry="2" />
    </svg>
  )
}

function TrainingIcon() {
  return (
    <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <polyline points="16 7 21 12 16 17" />
      <polyline points="8 7 3 12 8 17" />
      <line x1="3" y1="12" x2="21" y2="12" />
    </svg>
  )
}

function InferenceIcon() {
  return (
    <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <circle cx="12" cy="12" r="10" />
      <line x1="12" y1="2" x2="12" y2="22" />
      <line x1="2" y1="12" x2="22" y2="12" />
    </svg>
  )
}

function ModelsIcon() {
  return (
    <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <rect x="2" y="3" width="20" height="14" rx="2" />
      <line x1="8" y1="21" x2="16" y2="21" />
      <line x1="12" y1="17" x2="12" y2="21" />
      <line x1="8" y1="17" x2="16" y2="17" />
    </svg>
  )
}

function AnalyticsIcon() {
  return (
    <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <line x1="22" y1="11" x2="2" y2="22" />
      <polyline points="22 2 11 13 2 22" />
      <line x1="2" y1="22" x2="2" y2="2" />
      <polyline points="22 2 22 12" />
      <polyline points="22 22 12 22" />
    </svg>
  )
}

function SettingsIcon() {
  return (
    <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <circle cx="12" cy="12" r="3" />
      <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82-.33 2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a2 2 0 0 1 2.83 0 1.65 1.65 0 0 0 1.82-.33 1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33 2 2 0 0 1 2.83 0l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82.33a1.65 1.65 0 0 0-.33 1.82z" />
    </svg>
  )
}

interface HeaderProps {
  user?: { name: string; email: string }
  onLogout?: () => void
}

export const Header: React.FC<HeaderProps> = ({ user, onLogout }) => {
  return (
    <header className="flex items-center justify-between bg-white border-b border-slate-200 px-6 py-3 shadow-sm z-10">
      <div className="flex items-center gap-4">
        <button className="text-slate-500 hover:text-sky-500 focus:outline-none focus:ring-2 focus:ring-sky-500 rounded-md p-2 transition-all">
          <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <line x1="4" y1="12" x2="20" y2="12" />
            <circle cx="5" cy="12" r="1" />
            <circle cx="12" cy="12" r="1" />
            <circle cx="19" cy="12" r="1" />
          </svg>
        </button>
        <div className="hidden md:flex items-center gap-1">
          <span className="text-sm font-medium text-slate-600">NeuroMove</span>
          <span className="text-xs text-slate-400">•</span>
          <span className="text-sm font-medium text-slate-900">Enterprise MI-BCI Platform</span>
        </div>
      </div>
      <div className="flex items-center gap-3">
        <button className="text-slate-500 hover:text-red-600 focus:outline-none focus:ring-2 focus:ring-red-500 rounded-md p-2 transition-all" onClick={onLogout}>
          <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M9 21H5a2 2 0 01-2-2V5a2 2 0 012-2h4" />
            <polyline points="16 17 21 12 16 7" />
            <line x1="21" y1="12" x2="9" y2="12" />
          </svg>
        </button>
        <div className="flex items-center gap-2">
          <span className="hidden sm:block text-sm font-medium text-slate-700">{user?.name || 'Admin User'}</span>
          <div className="w-8 h-8 rounded-full bg-sky-100 flex items-center justify-center text-sky-600 text-sm font-bold">
            {user?.name?.charAt(0) || 'A'}
          </div>
        </div>
      </div>
    </header>
  )
}

interface PageContainerProps {
  children: ReactNode
  title: string
  subtitle?: string
  actions?: ReactNode
}

export const PageContainer: React.FC<PageContainerProps> = ({ children, title, subtitle, actions }) => {
  return (
    <div className="mb-6">
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 mb-4">
        <div>
          <div className="flex items-center gap-2 text-sm text-slate-500 uppercase tracking-wider font-medium mb-1">
            NeuroMove
          </div>
          <h1 className="text-2xl font-bold text-slate-900 leading-tight">{title}</h1>
          {subtitle && <p className="text-sm text-slate-500 mt-1">{subtitle}</p>}
        </div>
        {actions && <div className="flex gap-2">{actions}</div>}
      </div>
      <div className="bg-white border border-slate-200 rounded-lg shadow-sm p-6">{children}</div>
    </div>
  )
}

interface CardProps {
  children: ReactNode
  title?: string
  className?: string
}

export const Card: React.FC<CardProps> = ({ children, title, className }) => {
  return (
    <div className={`card ${className || ''}`}>
      {title && (
        <div className="card-header">
          <h3 className="text-lg font-semibold text-slate-900">{title}</h3>
        </div>
      )}
      <div className="card-body">{children}</div>
    </div>
  )
}

interface StatCardProps {
  value: string | number
  label: string
  trend?: string
  icon?: ReactNode
  trendDirection?: 'up' | 'down' | 'neutral'
}

export const StatCard: React.FC<StatCardProps> = ({ value, label, trend, icon, trendDirection }) => {
  const trendColor = trendDirection === 'up' ? 'text-emerald-600' : trendDirection === 'down' ? 'text-red-600' : 'text-slate-500'
  const trendIcon = trendDirection === 'up' ? <TrendUpIcon /> : trendDirection === 'down' ? <TrendDownIcon /> : <TrendFlatIcon />

  return (
    <div className="kpi-card">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-xs font-medium text-slate-500 uppercase tracking-wider">{label}</p>
          <p className="text-2xl font-bold text-slate-900 mt-1">{value}</p>
          {trend && (
            <p className={`text-xs font-medium mt-1 ${trendColor} flex items-center gap-1`}>
              {trendIcon} {trend}
            </p>
          )}
        </div>
        {icon && <div className="w-10 h-10 bg-sky-50 rounded-lg flex items-center justify-center text-sky-600">{icon}</div>}
      </div>
    </div>
  )
}

function TrendUpIcon() {
  return <svg className="w-3 h-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="22 7 13 12 22 17" /></svg>
}
function TrendDownIcon() {
  return <svg className="w-3 h-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="22 17 13 12 22 7" /></svg>
}
function TrendFlatIcon() {
  return <svg className="w-3 h-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><line x1="3" y1="12" x2="21" y2="12" /></svg>
}

export interface NavItemProps {
  label: string
  icon?: ReactNode
  path?: string
  active?: boolean
  onClick?: () => void
}

export const NavItem: React.FC<NavItemProps> = ({ label, icon, path, active = false, onClick }) => {
  const handleClick = () => {
    if (onClick) onClick()
    else if (path) window.location.href = path
  }

  return (
    <button
      onClick={handleClick}
      className={`flex items-center gap-3 px-3 py-2 rounded-md text-sm font-medium transition-colors ${
        active ? 'bg-sky-50 text-sky-700' : 'text-slate-600 hover:bg-slate-100 hover:text-sky-700'
      }`}
    >
      {icon && <span>{icon}</span>}
      <span>{label}</span>
    </button>
  )
}

interface UserMenuProps {
  user?: { name: string; email: string }
  onLogout: () => void
}

export const UserMenu: React.FC<UserMenuProps> = ({ user, onLogout }) => {
  return (
    <div className="relative inline-block">
      <button onClick={onLogout} className="flex items-center gap-2 px-2 py-1 rounded-md text-sm font-medium text-slate-600 hover:bg-slate-100 hover:text-slate-900 transition-colors">
        <div className="w-6 h-6 rounded-full bg-sky-100 flex items-center justify-center text-sky-600 text-xs font-bold">
          {user?.name?.charAt(0) || 'A'}
        </div>
        <span>{user?.name || 'Admin User'}</span>
        <svg className="w-3 h-3 text-slate-400 ml-1" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M19 9l-7 7-7-7" />
        </svg>
      </button>
    </div>
  )
}

interface BreadcrumbProps {
  items: { label: string; path?: string }[]
}

export const Breadcrumb: React.FC<BreadcrumbProps> = ({ items }) => {
  return (
    <nav className="flex items-center gap-2 text-xs text-slate-500 mb-4">
      {items.map((item, index) => (
        <React.Fragment key={item.path || item.label}>
          {index > 0 && <span className="mx-1">/</span>}
          {item.path ? (
            <a href={item.path} className="hover:text-sky-600 transition-colors">{item.label}</a>
          ) : (
            <span className="text-slate-700 font-medium">{item.label}</span>
          )}
        </React.Fragment>
      ))}
    </nav>
  )
}