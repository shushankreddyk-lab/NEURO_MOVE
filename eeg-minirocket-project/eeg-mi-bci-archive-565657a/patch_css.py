import re

with open('dashboard/app.py', 'r', encoding='utf-8') as f:
    content = f.read()

new_css = '''<style>
    /* === ENTERPRISE NEURO-TECH THEME (RECRUITER OPTIMIZED) === */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif !important;
    }

    /* Base App Styling */
    .stApp {
        background-color: #09090b;
        background-image: 
            radial-gradient(circle at 0% 0%, rgba(37, 99, 235, 0.04) 0%, transparent 50%),
            radial-gradient(circle at 100% 100%, rgba(139, 92, 246, 0.04) 0%, transparent 50%);
        background-attachment: fixed;
    }
    
    /* Clean Minimalist KPI Cards */
    .kpi-card {
        background: #18181b;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        border-color: rgba(255, 255, 255, 0.15);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: 600;
        color: #f8fafc;
        letter-spacing: -0.025em;
        margin: 0;
        line-height: 1.2;
    }
    .kpi-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        margin-top: 8px;
        font-weight: 500;
    }
    .kpi-sub {
        font-size: 0.75rem;
        color: #10b981;
        margin-top: 4px;
        font-weight: 400;
    }
    
    /* Elegant Glass Panels (No excessive 3D or neon) */
    .glass-panel {
        background: #18181b;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 32px;
        margin-bottom: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        transition: border-color 0.3s ease;
    }
    .glass-panel:hover {
        border-color: rgba(255, 255, 255, 0.12);
    }

    /* Subtle Hero Elements */
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: #27272a;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 999px;
        padding: 6px 16px;
        font-size: 0.75rem;
        font-weight: 500;
        color: #e2e8f0;
        margin-bottom: 24px;
    }
    .pulse-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #3b82f6;
        box-shadow: 0 0 8px #3b82f6;
    }
    .hero-title {
        font-size: 3rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.025em !important;
        color: #f8fafc !important;
        margin-bottom: 16px !important;
    }
    .neon-title {
        background: linear-gradient(to right, #60a5fa, #a78bfa);
        -webkit-background-clip: text; 
        background-clip: text; 
        color: transparent !important;
    }
    .hero-subtitle {
        font-size: 1.125rem;
        color: #94a3b8;
        font-weight: 400;
        line-height: 1.6;
        max-width: 800px;
    }

    /* Clean Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 32px;
        background: transparent;
        border-bottom: 1px solid rgba(255,255,255,0.1) !important;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        color: #71717a;
        font-weight: 500;
        font-size: 0.875rem;
        border: none !important;
        padding: 12px 0;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #e4e4e7;
    }
    .stTabs [aria-selected="true"] {
        color: #f8fafc !important;
        border-bottom: 2px solid #3b82f6 !important;
    }

    /* Professional Buttons */
    .stButton>button {
        background: #27272a !important;
        color: #f8fafc !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
        padding: 10px 20px !important;
        font-weight: 500 !important;
        transition: background 0.2s ease, border-color 0.2s ease !important;
    }
    .stButton>button:hover {
        background: #3f3f46 !important;
        border-color: rgba(255, 255, 255, 0.2) !important;
    }
    .stButton>button[kind="primary"] {
        background: #3b82f6 !important;
        color: #ffffff !important;
        border: 1px solid #2563eb !important;
    }
    .stButton>button[kind="primary"]:hover {
        background: #2563eb !important;
        border-color: #1d4ed8 !important;
    }

    /* Inputs & UI elements */
    .stSelectbox>div>div>div, .stTextInput>div>div>input {
        color: #f8fafc !important;
    }
    div[data-baseweb="select"] > div {
        background: #18181b !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 8px !important;
    }
    .stAlert {
        background: #18181b;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
    }
    hr { border-color: rgba(255,255,255,0.08) !important; }

    .prob-container {
        background: #18181b;
        padding: 24px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        margin-top: 10px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .osc-container {
        background: #18181b;
        padding: 18px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        margin-top: 10px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
</style>'''

pattern = re.compile(r'<style>.*?</style>', re.DOTALL)
new_content = pattern.sub(new_css, content)

with open('dashboard/app.py', 'w', encoding='utf-8') as f:
    f.write(new_content)
    
print("CSS patched successfully with prob-container")
