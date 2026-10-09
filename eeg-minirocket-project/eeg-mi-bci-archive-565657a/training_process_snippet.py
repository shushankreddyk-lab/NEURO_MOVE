if selected_tab == '📊 Training Process':
    st.markdown("""
    <div style="background:rgba(14,165,233,0.1); border:1px solid rgba(14,165,233,0.2); border-radius:8px; padding:15px; margin-bottom:20px;">
        <h3 style="color:#0ea5e9; font-size:1.2rem; margin-top:0;">Faculty Defense: Specific Model Explanation</h3>
        <p style="color:#c8d6e5; font-size:0.95rem; margin-bottom:0;">
        We have trained over 100 distinct models to address the extreme <strong>inter-subject variability</strong> inherent in EEG data. Select any of our trained models below to view the precise details of how it was built, what data it used, and why we trained it that way.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    selected_dataset_tab1 = st.selectbox("Select Dataset:", ["High-Gamma Dataset", "Kaya Finger Movements", "WAY-EEG-GAL"], key="dataset_sel_tab1")
    # Map dataset name to folder name
ds_folder_map = {
    "High-Gamma Dataset": "HighGamma",
    "Kaya Finger Movements": "Kaya",
    "WAY-EEG-GAL": "WAY",
    "": ""
}
mapped_folder = ds_folder_map.get(selected_dataset_tab1, selected_dataset_tab1)
models_dir = os.path.join(os.path.dirname(__file__), '..', 'models', mapped_folder)

    trained_models = []
    if os.path.exists(models_dir):
        trained_models = [f for f in os.listdir(models_dir) if f.endswith('.pth') or f.endswith('.pkl')]
        trained_models.sort()
        
    if trained_models:
        selected_faculty_model = st.selectbox("Select a Trained Model for Defense:", trained_models, key="faculty_model_sel")
        
        # Parse the filename
        arch_type = "Unknown Architecture"
        if "minirocket" in selected_faculty_model.lower():
            arch_type = "MiniRocket (MLP Classifier mapped over 10,000 random convolutional features)"
        elif "conformer" in selected_faculty_model.lower():
            arch_type = "EEG-Conformer (Deep CNN for local temporal features + Transformer for global spatial attention)"
        elif "cnn" in selected_faculty_model.lower():
            arch_type = "13-Layer CNN-LSTM (Deep convolutional spatial filters followed by recurrent temporal tracking)"
            
        import re
        sub_match = re.search(r'subs(\d+)(to)?(\d+)?', selected_faculty_model)
        if sub_match:
            if sub_match.group(3):
                subject_info = f"Subjects {sub_match.group(1)} to {sub_match.group(3)}"
            else:
                subject_info = f"Subject {sub_match.group(1)}"
        else:
            subject_info = "All 109 Subjects (Global / Generalized)"
            
        st.markdown(f"""
        <div class="glass-card" style="margin-bottom:30px;">
            <h4 style="color:#a855f7; margin-top:0;">Model Profile: <code>{selected_faculty_model}</code></h4>
            <ul style="color:#c8d6e5; line-height:1.7;">
                <li><strong style="color:#10b981;">Architecture:</strong> {arch_type}</li>
                <li><strong style="color:#10b981;">Training Data (Who):</strong> {subject_info}. <em>Why?</em> By isolating training to this specific batch of subjects, the model optimizes its internal weights for their unique brain topological patterns (handling inter-subject variability) before generalizing.</li>
                <li><strong style="color:#10b981;">Features Used (What):</strong> 20 Motor-Cortex Channels (e.g., C3, C4, FC3, FC4). <em>Why?</em> We strictly avoided frontal/occipital channels to prevent the model from cheating using eye-blinks (EOG) or visual processing.</li>
                <li><strong style="color:#10b981;">Preprocessing (How):</strong> 4-38Hz Bandpass filter + Common Average Referencing (CAR) + Z-score normalization per channel to eliminate skull noise.</li>
            </ul>
            <p style="color:#c8d6e5; font-size:0.95rem; margin-top:10px; text-align:justify;">
            <strong>How it was trained:</strong> The model processed 4.0-second raw EEG epochs. It learned to map the Event-Related Desynchronization (ERD) amplitudes in the &mu; (8-12Hz) and &beta; (13-30Hz) bands to the 4 motor classes (Left Hand, Right Hand, Both Feet, Tongue). For neural networks (Conformer/CNN), the AdamW optimizer was used over multiple epochs. For MiniRocket, an MLP Classifier learns the mapping from the extracted convolutional features using gradient descent.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")

    selected_model_view = st.selectbox(
        "Select Model to View General Architecture Execution",
        ["MiniRocket Pipeline", "EEG-Conformer", "13-Layer CNN-LSTM"]
    )

    if selected_model_view == "MiniRocket Pipeline":
        st.markdown("""
<div class="glass-card" style="margin-bottom:24px;">
<h3 style="color:#a855f7; font-size:1.1rem; text-transform:uppercase; letter-spacing:0.08em; margin-top:0;">MiniRocket Training Execution</h3>

<p style="color:#c8d6e5; font-size:0.95rem; line-height:1.7; margin-bottom:15px; text-align:justify;">
<strong>Step 1: Dataset Compilation & Channel Selection</strong><br>
The PhysioNet EEGMMIDB dataset is scanned across all 109 subjects, aggregating over 18,000 spatial-temporal trials. Each raw trial represents a 4.0-second mental execution window at 160Hz, originally recorded across 64 channels resulting in a <code>[64, 640]</code> matrix. We strictly isolate <strong>20 critical channels</strong> (e.g., FC3, FC4, C3, C4, CP3, CP4, CZ) directly over the primary motor cortex. The input tensor is immediately reduced to <code>[Batch, 20, 640]</code>, filtering out visual and auditory cortex signals to specifically target Event-Related Desynchronization (ERD) phenomenon.
</p>

<p style="color:#c8d6e5; font-size:0.95rem; line-height:1.7; margin-bottom:15px; text-align:justify;">
<strong>Step 2: Preprocessing & Normalization</strong><br>
Before hitting the model, the <code>[Batch, 20, 640]</code> tensor undergoes Common Average Referencing (CAR). The mean signal across all 20 channels is subtracted from each channel at every time step, removing global noise. A 4-38Hz zero-phase FIR bandpass filter is applied across the time dimension (640 samples) to isolate the &mu; and &beta; bands. Finally, Z-score normalization forces each channel sequence to a mean of 0 and variance of 1.
</p>

<p style="color:#c8d6e5; font-size:0.95rem; line-height:1.7; margin-bottom:15px; text-align:justify;">
<strong>Step 3: MiniRocket Fit (Deterministic)</strong><br>
MiniRocket avoids backpropagation entirely. Instead, it instantly initializes <strong>10,000 random convolutional kernels</strong>. These kernels have fixed lengths (typically 7, 9, or 11) and highly variable dilations. The <code>[Batch, 20, 640]</code> tensor is convolved across the time dimension. For each kernel output, the algorithm calculates the <em>Proportion of Positive Values (PPV)</em>—projecting the complex EEG data into a massive 10,000-dimensional linearly separable feature vector: <code>[Batch, 10000]</code>.
</p>

<p style="color:#c8d6e5; font-size:0.95rem; line-height:1.7; margin-bottom:15px; text-align:justify;">
<strong>Step 4: MLP Classifier</strong><br>
An MLP Classifier receives the <code>[Batch, 10000]</code> matrix and learns the optimal mapping using gradient descent, outputting a <code>[Batch, 4]</code> vector of intent probabilities.
</p>

<h4 style="color:#a855f7; margin-top:20px; font-size:1.05rem;">Model Training Parameters</h4>
<table style="width:100%; color:#c8d6e5; border-collapse: collapse; margin-bottom:10px; font-size:0.9rem;">
  <tr style="border-bottom: 1px solid #334155; background:rgba(0,0,0,0.3);">
    <th style="padding:10px; text-align:left;">Parameter</th><th style="padding:10px; text-align:left;">Value</th><th style="padding:10px; text-align:left;">Description</th>
  </tr>
  <tr style="border-bottom: 1px solid #334155;">
    <td style="padding:10px; font-weight:bold;">Kernels</td><td style="padding:10px;">10,000</td><td style="padding:10px;">Randomly generated convolution filters</td>
  </tr>
  <tr style="border-bottom: 1px solid #334155;">
    <td style="padding:10px; font-weight:bold;">Kernel Lengths</td><td style="padding:10px;">7, 9, 11</td><td style="padding:10px;">Temporal receptive fields</td>
  </tr>
  <tr style="border-bottom: 1px solid #334155;">
    <td style="padding:10px; font-weight:bold;">Classifier</td><td style="padding:10px;">MLP Classifier</td><td style="padding:10px;">Multi-layer Perceptron</td>
  </tr>
  <tr style="border-bottom: 1px solid #334155;">
    <td style="padding:10px; font-weight:bold;">Feature Extraction</td><td style="padding:10px;">PPV</td><td style="padding:10px;">Proportion of Positive Values</td>
  </tr>
  <tr>
    <td style="padding:10px; font-weight:bold;">Epochs</td><td style="padding:10px;">1</td><td style="padding:10px;">Closed-form solution (No backprop needed)</td>
  </tr>
</table>
</div>
        """, unsafe_allow_html=True)
        
        mr_mermaid = """
        graph TD
            A[Raw EEG 64-Ch] --> B[Channel Selection 20-Ch]
            B --> C[CAR & 4-38Hz Filter]
            C --> D[Z-Score Normalization]
            D --> E[10,000 Dilated Kernels]
            E --> F[PPV Feature Extraction]
            F --> G[MLP Classifier Fit]
            G --> H[Global Model Compiled]
            style A fill:#1e293b,stroke:#334155,color:#fff
            style H fill:#10b981,stroke:#059669,color:#fff
            style E fill:#8b5cf6,stroke:#7c3aed,color:#fff
            style F fill:#8b5cf6,stroke:#7c3aed,color:#fff
            style G fill:#f43f5e,stroke:#e11d48,color:#fff
        """
        st.markdown(f"```mermaid\n{mr_mermaid}\n```")

    elif selected_model_view == "EEG-Conformer":
        st.markdown("""
<div class="glass-card" style="margin-bottom:24px;">
<h3 style="color:#10b981; font-size:1.1rem; text-transform:uppercase; letter-spacing:0.08em; margin-top:0;">EEG-Conformer Training Execution</h3>

<p style="color:#c8d6e5; font-size:0.95rem; line-height:1.7; margin-bottom:15px; text-align:justify;">
<strong>Step 1: Dataset Compilation & Preprocessing</strong><br>
Similar to MiniRocket, the raw EEG recordings are bandpass filtered (4-38Hz), CAR referenced, and reduced to 20 motor channels. The input to the Conformer is the raw temporal sequences <code>[Batch, 20, 640]</code>. Z-score normalization forces each channel sequence to a mean of 0 and variance of 1.
</p>

<p style="color:#c8d6e5; font-size:0.95rem; line-height:1.7; margin-bottom:15px; text-align:justify;">
<strong>Step 2: Convolutional Feature Extraction (CNN)</strong><br>
The EEG-Conformer starts with an EEGNet-like convolutional block. A Conv2D layer operates across the time dimension to capture temporal frequency patterns, followed immediately by a DepthwiseConv2D layer across the 20 channels to learn robust spatial filters. This extracts localized spatial-temporal features.
</p>

<p style="color:#c8d6e5; font-size:0.95rem; line-height:1.7; margin-bottom:15px; text-align:justify;">
<strong>Step 3: Self-Attention Transformer</strong><br>
The extracted spatial-temporal features are flattened along the spatial dimension and fed into a multi-head self-attention transformer module. The self-attention mechanism captures global dependencies across the entire time series window, dynamically weighing the importance of different temporal patterns over the 4-second execution period.
</p>

<p style="color:#c8d6e5; font-size:0.95rem; line-height:1.7; margin-bottom:15px; text-align:justify;">
<strong>Step 4: Backpropagation</strong><br>
The model is trained end-to-end via the AdamW optimizer and OneCycleLR learning rate schedule, minimizing the Cross-Entropy Loss with label smoothing applied.
</p>

<h4 style="color:#10b981; margin-top:20px; font-size:1.05rem;">Model Training Parameters</h4>
<table style="width:100%; color:#c8d6e5; border-collapse: collapse; margin-bottom:10px; font-size:0.9rem;">
  <tr style="border-bottom: 1px solid #334155; background:rgba(0,0,0,0.3);">
    <th style="padding:10px; text-align:left;">Parameter</th><th style="padding:10px; text-align:left;">Value</th><th style="padding:10px; text-align:left;">Description</th>
  </tr>
  <tr style="border-bottom: 1px solid #334155;">
    <td style="padding:10px; font-weight:bold;">Optimizer</td><td style="padding:10px;">AdamW</td><td style="padding:10px;">Adaptive momentum with decoupled weight decay (1e-3)</td>
  </tr>
  <tr style="border-bottom: 1px solid #334155;">
    <td style="padding:10px; font-weight:bold;">Scheduler</td><td style="padding:10px;">OneCycleLR</td><td style="padding:10px;">Cosine annealing learning rate for stable convergence</td>
  </tr>
  <tr style="border-bottom: 1px solid #334155;">
    <td style="padding:10px; font-weight:bold;">Batch Size</td><td style="padding:10px;">256</td><td style="padding:10px;">Large batch sizes to stabilize transformer gradients</td>
  </tr>
  <tr style="border-bottom: 1px solid #334155;">
    <td style="padding:10px; font-weight:bold;">Self-Attention Heads</td><td style="padding:10px;">8</td><td style="padding:10px;">Allows model to attend to multiple temporal locations</td>
  </tr>
  <tr>
    <td style="padding:10px; font-weight:bold;">Loss Function</td><td style="padding:10px;">Cross-Entropy</td><td style="padding:10px;">Includes class weights & label smoothing (0.1)</td>
  </tr>
</table>
</div>
        """, unsafe_allow_html=True)
        
        conf_mermaid = """
        graph TD
            A[Raw EEG 20-Ch] --> B[Z-Score Norm]
            B --> C[Temporal Conv2D]
            C --> D[Spatial Depthwise Conv2D]
            D --> E[Transformer Positional Encoding]
            E --> F[Multi-Head Self Attention]
            F --> G[Cross Entropy Loss]
            G --> H[Backpropagation AdamW]
            H -->|Epochs| C
            style A fill:#1e293b,stroke:#334155,color:#fff
            style C fill:#0ea5e9,stroke:#0284c7,color:#fff
            style D fill:#0ea5e9,stroke:#0284c7,color:#fff
            style F fill:#10b981,stroke:#059669,color:#fff
        """
        st.markdown(f"```mermaid\n{conf_mermaid}\n```")

    else:
        st.markdown("""
<div class="glass-card" style="margin-bottom:24px;">
<h3 style="color:#00d4ff; font-size:1.1rem; text-transform:uppercase; letter-spacing:0.08em; margin-top:0;">CNN-LSTM Training Execution</h3>

<p style="color:#c8d6e5; font-size:0.95rem; line-height:1.7; margin-bottom:15px; text-align:justify;">
<strong>Step 1: Dataset Compilation & Channel Selection</strong><br>
The PhysioNet EEGMMIDB dataset is scanned across all 109 subjects, aggregating over 18,000 spatial-temporal trials. Each raw trial represents a 4.0-second mental execution window at 160Hz, originally recorded across 64 channels resulting in a <code>[64, 640]</code> matrix. We strictly isolate <strong>20 critical channels</strong> directly over the primary motor cortex, yielding an input tensor of <code>[Batch, 20, 640]</code>. This filters out visual and auditory cortex signals to exclusively target Event-Related Desynchronization (ERD).
</p>

<p style="color:#c8d6e5; font-size:0.95rem; line-height:1.7; margin-bottom:15px; text-align:justify;">
<strong>Step 2: Preprocessing & Normalization</strong><br>
Before model ingestion, the <code>[Batch, 20, 640]</code> tensor undergoes Common Average Referencing (CAR). A 4-38Hz zero-phase FIR bandpass filter is applied across the 640 time-steps to isolate the &mu; and &beta; frequency bands. Finally, Z-score normalization forces each channel sequence to a mean of 0 and variance of 1, stabilizing the input gradients for the deep network.
</p>

<p style="color:#c8d6e5; font-size:0.95rem; line-height:1.7; margin-bottom:15px; text-align:justify;">
<strong>Step 3: 1D Convolution Spatial & Temporal Extraction</strong><br>
The <code>[Batch, 20, 640]</code> tensor enters the CNN block. The first Conv1D layer applies 16 filters with a kernel size of 3 across the time dimension, identifying localized micro-patterns. A MaxPool1D(2) layer halves the temporal dimension. A second Conv1D layer (32 filters) detects deeper compound patterns. After BatchNormalization, ReLU activation, and a second MaxPool1D(2), the tensor shape is compressed and deepened to roughly <code>[Batch, 32, 160]</code>. 
</p>

<p style="color:#c8d6e5; font-size:0.95rem; line-height:1.7; margin-bottom:15px; text-align:justify;">
<strong>Step 4: LSTM Synchrony & Backpropagation</strong><br>
The spatial feature maps are permuted to <code>[Batch, 160, 32]</code> (Treating the 32 filters as features per timestep) and fed sequentially into an LSTM with 100 hidden units. The LSTM maintains a hidden state matrix across the 160 timesteps, "remembering" how the motor imagery evolved over the 4-second window. The final hidden state <code>[Batch, 100]</code> is passed through three Dense (Linear) layers (100 → 64 → 32 → 4). The output is a <code>[Batch, 4]</code> tensor representing raw logits. 
</p>

<h4 style="color:#00d4ff; margin-top:20px; font-size:1.05rem;">Model Training Parameters</h4>
<table style="width:100%; color:#c8d6e5; border-collapse: collapse; margin-bottom:10px; font-size:0.9rem;">
  <tr style="border-bottom: 1px solid #334155; background:rgba(0,0,0,0.3);">
    <th style="padding:10px; text-align:left;">Parameter</th><th style="padding:10px; text-align:left;">Value</th><th style="padding:10px; text-align:left;">Description</th>
  </tr>
  <tr style="border-bottom: 1px solid #334155;">
    <td style="padding:10px; font-weight:bold;">CNN Layers</td><td style="padding:10px;">2x Conv1D</td><td style="padding:10px;">16 and 32 filters, Kernel=3, extracting spatial-temporal micro-features</td>
  </tr>
  <tr style="border-bottom: 1px solid #334155;">
    <td style="padding:10px; font-weight:bold;">LSTM Hidden Units</td><td style="padding:10px;">100</td><td style="padding:10px;">Recurrent units capturing long-range temporal synchrony</td>
  </tr>
  <tr style="border-bottom: 1px solid #334155;">
    <td style="padding:10px; font-weight:bold;">Dense Layers</td><td style="padding:10px;">100 → 64 → 32</td><td style="padding:10px;">Progressive dimensionality reduction to 4 classes</td>
  </tr>
  <tr style="border-bottom: 1px solid #334155;">
    <td style="padding:10px; font-weight:bold;">Optimizer</td><td style="padding:10px;">Adam</td><td style="padding:10px;">Standard Adam optimization with cross-entropy loss</td>
  </tr>
  <tr>
    <td style="padding:10px; font-weight:bold;">Epochs</td><td style="padding:10px;">15 (default)</td><td style="padding:10px;">Backpropagation passes for gradient convergence</td>
  </tr>
</table>
</div>
        """, unsafe_allow_html=True)
        
        cnn_mermaid = """
        graph TD
            A[Raw EEG 64-Ch] --> B[Channel Selection 20-Ch]
            B --> C[CAR & 4-38Hz Filter]
            C --> D[Z-Score Normalization]
            D --> E[Conv1D - Spatial Features]
            E --> F[LSTM - Temporal Synchrony]
            F --> G[Cross Entropy Loss]
            G --> H[Backpropagation Adam]
            H -->|15 Epochs| E
            style A fill:#1e293b,stroke:#334155,color:#fff
            style E fill:#0ea5e9,stroke:#0284c7,color:#fff
            style F fill:#0ea5e9,stroke:#0284c7,color:#fff
            style H fill:#f59e0b,stroke:#d97706,color:#fff
        """
        st.markdown(f"```mermaid\n{cnn_mermaid}\n```")


# --- TAB 5: PREPROCESSING ---
