import sys
import os
import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import PHYSIONET_CLASSES
from src.minirocket_pipeline import MiniRocketPipeline

st.set_page_config(
    page_title="MiniRocket EEG Classification Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------
# 1. VISUAL STYLING & POLISH (DARK MODE + GLASSMORPHISM)
# -------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    .stApp { background-color: #0B0F19; color: #E2E8F0; }

    h1[id], h2[id], h3[id] { scroll-margin-top: 80px; }

    .glass-card {
        background: rgba(30, 41, 59, 0.4);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 30px -10px rgba(6, 182, 212, 0.2);
        border: 1px solid rgba(6, 182, 212, 0.3);
    }

    .neon-text {
        background: linear-gradient(90deg, #06b6d4, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.8rem;
        margin-bottom: 0.5rem;
    }
    
    .sub-neon { color: #94A3B8; font-size: 1.1rem; font-weight: 400; margin-bottom: 2rem; }

    .kpi-box { text-align: center; padding: 15px; }
    .kpi-title { font-size: 0.9rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; }
    .kpi-value { font-size: 2rem; font-weight: 700; color: #F8FAFC; }
    .kpi-cyan { color: #22D3EE; }
    .kpi-purple { color: #C084FC; }

    .branch-title { font-size: 1.3rem; font-weight: 600; margin-bottom: 15px; border-bottom: 2px solid; padding-bottom: 10px; }
    .branch-a-title { border-color: #06b6d4; color: #06b6d4; }
    .branch-b-title { border-color: #a855f7; color: #a855f7; }

    .stButton>button {
        background: linear-gradient(90deg, #06b6d4, #a855f7) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px !important;
        transition: opacity 0.2s ease !important;
        box-shadow: 0 4px 14px 0 rgba(6, 182, 212, 0.39) !important;
    }
    .stButton>button:hover { opacity: 0.85 !important; }
    
    .block-container { padding-top: 2rem; }
    a { color: #22D3EE; text-decoration: none; }
    a:hover { text-decoration: underline; }
    
    .nav-link {
        display: block; padding: 10px 15px; color: #CBD5E1; text-decoration: none;
        border-radius: 8px; margin-bottom: 5px; transition: background 0.2s ease, color 0.2s ease;
    }
    .nav-link:hover { background: rgba(255, 255, 255, 0.1); color: #fff; text-decoration: none; }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# SIDEBAR NAVIGATION
# -------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🧭 Navigation")
    st.markdown("""
        <a href="#section-1-hero-performance-dashboard" class="nav-link">🏠 Hero & KPI Dashboard</a>
        <a href="#section-2-dual-stream-architectural-breakdown" class="nav-link">🧬 Architecture Breakdown</a>
        <a href="#section-3-interactive-signal-inference-lab" class="nav-link">🔬 Interactive Inference Lab</a>
        <a href="#section-4-base-research-paper-showcase" class="nav-link">📚 Research Publication</a>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### ⚙️ Engine Status")
    st.success("✅ Real Models Loaded")
    st.success("✅ Real Test Data Loaded")

# -------------------------------------------------------------
# CACHED DATA & MODEL LOADING
# -------------------------------------------------------------
@st.cache_resource(show_spinner="Loading Real Models & Test Data...")
def load_real_assets():
    model_dir = Path("models")
    npz_path = model_dir / "sample_test_data_physionet.npz"
    model_path = model_dir / "minirocket_physionet_s01.joblib"
    
    if not npz_path.exists() or not model_path.exists():
        st.error("Model artifacts missing. Run `python src/train.py` first.")
        st.stop()
        
    mr_pipeline = MiniRocketPipeline.load(model_path)
    data = np.load(npz_path)
    X_test = data['X_test']
    y_test = data['y_test']
    ch_names = list(data['ch_names'])
    
    num_samples = min(20, len(X_test))
    return mr_pipeline, X_test[:num_samples], y_test[:num_samples], ch_names

@st.cache_data
def load_metrics():
    try:
        df = pd.read_csv("results/metrics/benchmark_summary_physionet.csv")
        mr_acc = df["minirocket_acc"].mean() * 100
        mr_lat = df["minirocket_latency_ms"].mean()
    except Exception:
        mr_acc, mr_lat = 0.0, 0.0
    return mr_acc, mr_lat

mr_model, X_test_bank, y_test_bank, ch_names = load_real_assets()
mr_acc, mr_lat = load_metrics()

# Session State for tracking inference results
if "correct_count" not in st.session_state:
    st.session_state.correct_count = 0
if "total_count" not in st.session_state:
    st.session_state.total_count = 0
if "tested_trials" not in st.session_state:
    st.session_state.tested_trials = set()

# -------------------------------------------------------------
# SECTION 1: HERO & PERFORMANCE DASHBOARD
# -------------------------------------------------------------
st.markdown("<div id='section-1-hero-performance-dashboard'></div>", unsafe_allow_html=True)
st.markdown("<div class='neon-text'>Motor Imagery EEG Signal Classification</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-neon'>Ultra-Fast BCI Inference using Minimally Random Convolutional Kernel Transform (MiniRocket) vs Hybrid Deep Learning</div>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown(f"""
    <div class="glass-card">
        <h3 class="branch-title branch-a-title">🚀 MiniRocket Branch (Trained)</h3>
        <div style="display: flex; justify-content: space-around;">
            <div class="kpi-box">
                <div class="kpi-title">Actual Accuracy</div>
                <div class="kpi-value kpi-cyan">{mr_acc:.2f}%</div>
            </div>
            <div class="kpi-box">
                <div class="kpi-title">Actual Latency</div>
                <div class="kpi-value kpi-cyan">{mr_lat:.2f} ms</div>
            </div>
            <div class="kpi-box">
                <div class="kpi-title">Parameters</div>
                <div class="kpi-value kpi-cyan">~20k</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="glass-card">
        <h3 class="branch-title branch-b-title">🧠 CNN-LSTM Branch (Reference)</h3>
        <div style="display: flex; justify-content: space-around;">
            <div class="kpi-box">
                <div class="kpi-title">Reported Accuracy</div>
                <div class="kpi-value kpi-purple">98.06%</div>
            </div>
            <div class="kpi-box">
                <div class="kpi-title">Reported Latency</div>
                <div class="kpi-value kpi-purple">8.0 ms</div>
            </div>
            <div class="kpi-box">
                <div class="kpi-title">Parameters</div>
                <div class="kpi-value kpi-purple">~250k</div>
            </div>
        </div>
        <p style="text-align:center; color:#94A3B8; font-size:12px; margin-top:-10px;">* Reference: Base Paper Reported Result (Not actively trained)</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)

# -------------------------------------------------------------
# SECTION 2: DUAL-STREAM ARCHITECTURAL BREAKDOWN
# -------------------------------------------------------------
st.markdown("<div id='section-2-dual-stream-architectural-breakdown'></div>", unsafe_allow_html=True)
st.markdown("## 🧬 Dual-Stream Architectural Breakdown")

colA, colB = st.columns(2)
with colA:
    st.markdown("""
    <div class="glass-card" style="height: 100%;">
        <h4 style="color: #22D3EE;">Branch A: MiniRocket + RidgeClassifierCV</h4>
        <p style="color: #94A3B8;">A lightweight, deterministic transformation pipeline offering state-of-the-art accuracy with minimal computational overhead.</p>
        <ul style="color: #CBD5E1; font-weight: 300; line-height: 1.8;">
            <li><b>Feature Extraction:</b> 10,000 fixed, random dilated convolutional kernels.</li>
            <li><b>Pooling:</b> Proportion of Positive Values (PPV) geometrically captures temporal motifs without expensive gradient descent.</li>
            <li><b>Classification:</b> Closed-form <code>RidgeClassifierCV</code> with L2 regularization solving convex optimization instantly.</li>
            <li><b>Latency Profile:</b> Extreme speedup (~13x) enabling embedded mobile BCI deployment.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with colB:
    st.markdown("""
    <div class="glass-card" style="height: 100%;">
        <h4 style="color: #C084FC;">Branch B: Hybrid CNN-LSTM</h4>
        <p style="color: #94A3B8;">A high-capacity deep learning baseline that leverages multi-scale spatial convolution followed by recurrent temporal modeling.</p>
        <ul style="color: #CBD5E1; font-weight: 300; line-height: 1.8;">
            <li><b>Spatial Feature Extraction:</b> Two 1D Convolutional blocks (k=25, 13) extracting spatial representations across 64 channels.</li>
            <li><b>Temporal Modeling:</b> 2-layer stacked Long Short-Term Memory (LSTM) network tracking trial-wise state dynamics.</li>
            <li><b>Regularization:</b> BatchNormalization, Dropout (0.5), and L2 Weight Decay (Adam Optimizer) to prevent overfitting.</li>
            <li><b>Latency Profile:</b> High-capacity but slower execution, requiring GPU acceleration for real-time operation.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)

# -------------------------------------------------------------
# SECTION 3: INTERACTIVE SIGNAL & INFERENCE LAB
# -------------------------------------------------------------
st.markdown("<div id='section-3-interactive-signal-inference-lab'></div>", unsafe_allow_html=True)
st.markdown("## 🔬 Interactive Signal & Inference Lab")
st.markdown("<p style='color: #94A3B8;'>Run the trained MiniRocket model on <b>real, held-out EEG trials</b> it has never seen during training, and compare predictions to the ground truth.</p>", unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    c1, c2 = st.columns([2, 1])
    with c1:
        selected_trial_idx = st.selectbox(
            "🧪 Select Real Test Trial (Hold-out Data)",
            options=list(range(len(y_test_bank))),
            format_func=lambda x: f"Test Trial #{x+1}"
        )
    with c2:
        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("Run Live Decode 🚀", width='stretch')

    if run_btn:
        single_trial = X_test_bank[selected_trial_idx:selected_trial_idx+1]
        true_intent = y_test_bank[selected_trial_idx]

        # Plot Waveforms
        plt.style.use('dark_background')
        fig, axes = plt.subplots(3, 1, figsize=(10, 5), sharex=True, facecolor='#1E293B')
        time_vec = np.linspace(-0.5, 4.0, single_trial.shape[2])
        
        plot_channels = ['C3', 'Cz', 'C4']
        colors = ['#06b6d4', '#f59e0b', '#a855f7']
        
        for ax, ch, color in zip(axes, plot_channels, colors):
            ax.set_facecolor('#1E293B')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#475569')
            ax.spines['bottom'].set_color('#475569')
            ax.tick_params(colors='#94A3B8')
            
            idx = ch_names.index(ch) if ch in ch_names else 0
            ax.plot(time_vec, single_trial[0, idx, :], color=color, lw=1.5, label=f"Ch {ch}")
            ax.axvline(0.0, color='#EF4444', linestyle='--', lw=1, alpha=0.8)
            ax.axvspan(0.0, 3.5, color='#FDE68A', alpha=0.1) 
            ax.legend(loc='upper right', frameon=False, labelcolor='white')
            
        axes[-1].set_xlabel("Time (s)", color='#94A3B8')
        fig.tight_layout()
        st.pyplot(fig)

        st.markdown("---")
        
        # Inference
        t0 = time.perf_counter()
        mr_probs = mr_model.predict_proba(single_trial)[0]
        mr_pred = int(np.argmax(mr_probs))
        mr_time = (time.perf_counter() - t0) * 1000

        is_correct = (mr_pred == true_intent)
        if selected_trial_idx not in st.session_state.tested_trials:
            st.session_state.tested_trials.add(selected_trial_idx)
            st.session_state.total_count += 1
            if is_correct:
                st.session_state.correct_count += 1

        r1, r2 = st.columns([1, 1])
        with r1:
            st.markdown(f"#### 🚀 MiniRocket Real Inference")
            st.markdown(f"**Predicted Intent:** `{PHYSIONET_CLASSES.get(mr_pred, 'Unknown')}`")
            st.markdown(f"**Actual Intent (Ground Truth):** `{PHYSIONET_CLASSES.get(true_intent, 'Unknown')}`")
            match_str = "<span style='color:#10B981; font-weight: bold;'>Match ✅</span>" if is_correct else "<span style='color:#EF4444; font-weight: bold;'>Miss ❌</span>"
            st.markdown(f"**Result:** {match_str}", unsafe_allow_html=True)
            st.markdown(f"**Latency:** `{mr_time:.2f} ms`")
            mr_conf = min(1.0, max(0.0, float(np.max(mr_probs))))
            st.progress(mr_conf)
            st.caption(f"Confidence: {mr_conf*100:.1f}%")
            
        with r2:
            st.markdown(f"#### 📊 Session Performance")
            st.info(f"**{st.session_state.correct_count} out of {st.session_state.total_count} trials correctly classified in this session**")

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)

# -------------------------------------------------------------
# SECTION 4: BASE RESEARCH PAPER SHOWCASE
# -------------------------------------------------------------
st.markdown("<div id='section-4-base-research-paper-showcase'></div>", unsafe_allow_html=True)
st.markdown("## 📚 Research Publication")

st.markdown("""
<div class="glass-card">
    <h3 style="color: white; margin-bottom: 5px;">Motor imagery EEG signal classification using minimally random convolutional kernel transform and hybrid deep learning</h3>
    <p style="color: #06b6d4; font-weight: 600;">Jamal Hwaidi, Mohamed Chahine Ghanem</p>
    <p style="color: #94A3B8; font-style: italic;">NeuroImage, Volume 328 (2026) 121816</p>
    <hr style="border-color: #334155;">
    
    <h4 style="color: #E2E8F0;">Abstract Summary</h4>
    <p style="color: #CBD5E1; line-height: 1.6;">
    Real-time decoding of motor imagery from electroencephalography (EEG) is a fundamental challenge in 
    Brain-Computer Interfaces (BCI). Deep learning approaches, particularly hybrid architectures like CNN-LSTMs, 
    have driven recent accuracy improvements but impose heavy computational overhead, bottlenecking mobile 
    clinical neurorehabilitation. This study presents a head-to-head evaluation of the <b>MiniRocket</b> 
    transform against a state-of-the-art hybrid CNN-LSTM. 
    </p>
    
    <div style="margin-top: 25px; padding: 15px; border-left: 4px solid #C084FC; background: rgba(192, 132, 252, 0.1); border-radius: 4px;">
        <h4 style="margin: 0 0 10px 0; color: #C084FC;">💡 Clinical Impact highlight</h4>
        <p style="margin: 0; color: #E2E8F0; line-height: 1.5;">
        The ~0.6 ms inference latency provided by MiniRocket unlocks the deployment of highly accurate, high-channel count BCI decoders directly onto embedded, low-power microcontrollers (such as ARM Cortex-M or RISC-V). This fundamentally enables untethered, mobile exoskeleton neurorehabilitation for stroke patients without requiring cloud connectivity or heavy gaming GPUs.
        </p>
    </div>
</div>
""", unsafe_allow_html=True)
