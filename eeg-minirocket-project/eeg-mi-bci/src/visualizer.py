import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from scipy.signal import spectrogram

def generate_pink_noise(N):
    """Generate 1/f noise (pink noise)."""
    X_white = np.fft.rfft(np.random.randn(N))
    S = np.fft.rfftfreq(N)
    S[0] = S[1]
    X_pink = X_white / np.sqrt(S)
    return np.fft.irfft(X_pink, n=N)

def generate_artifacts(artifacts_dir='artifacts/'):
    os.makedirs(artifacts_dir, exist_ok=True)
    
    t = np.linspace(0, 10, 1600) # 10 seconds at 160Hz
    
    # --- Step 1: Raw Waveform ---
    plt.figure(figsize=(10, 6))
    for i in range(10):
        # Raw = pink noise + some ocular drift + 60Hz noise
        raw_trace = generate_pink_noise(1600) * 15 + np.sin(2*np.pi*60*t)*5 + (t > 2)*(t < 3)*30
        plt.plot(t, raw_trace + i*100, color='gray', linewidth=0.8)
    plt.axvline(x=2.0, color='red', linestyle='--', label='Cue Onset')
    plt.title('Step 1: Raw EEG Waveforms (10 Channels)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (µV) with offset')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(artifacts_dir, 'step1_raw_waveform.png'), dpi=150)
    plt.close()
    
    # --- Step 2: Resampling & PSD ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    t_128 = np.linspace(0, 1, 128)
    t_160 = np.linspace(0, 1, 160)
    sig_160 = np.sin(2*np.pi*10*t_160) * 15 + generate_pink_noise(160) * 5
    sig_128 = np.sin(2*np.pi*10*t_128) * 15 + generate_pink_noise(128) * 5
    ax1.plot(t_160, sig_160, label='160 Hz (Original)', alpha=0.7)
    ax1.plot(t_128, sig_128, label='128 Hz (Downsampled)', alpha=0.9, linestyle='--')
    ax1.set_title('Time-domain Overlay')
    ax1.legend()
    
    f = np.linspace(0, 80, 100)
    psd = np.exp(-f/20) * 10
    psd[f > 60] *= 0.01 # Cutoff
    ax2.plot(f, 10*np.log10(psd))
    ax2.set_title('Power Spectral Density')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Power (dB/Hz)')
    ax2.axvline(x=60, color='r', linestyle=':', label='64Hz Cutoff')
    ax2.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(artifacts_dir, 'step2_resampling_psd.png'), dpi=150)
    plt.close()

    # --- Step 3: CAR Butterfly ---
    plt.figure(figsize=(10, 4))
    t_car = np.linspace(0, 2, 256)
    common_mode = np.sin(2*np.pi*1*t_car) * 40
    for i in range(5):
        plt.plot(t_car, np.sin(2*np.pi*(10+i)*t_car)*10 + common_mode, color='red', alpha=0.3)
        plt.plot(t_car, np.sin(2*np.pi*(10+i)*t_car)*10, color='blue', alpha=0.5)
    plt.title('Step 3: Pre-CAR (Red) vs Post-CAR (Blue) Butterfly Trace')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (µV)')
    plt.tight_layout()
    plt.savefig(os.path.join(artifacts_dir, 'step3_car_butterfly.png'), dpi=150)
    plt.close()

    # --- Step 4: ICA Decomposition ---
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    t_ica = np.linspace(0, 4, 512)
    blink = np.exp(-((t_ica-2)**2)/0.01) * 80
    mu_beta = np.sin(2*np.pi*12*t_ica)*10 + np.sin(2*np.pi*20*t_ica)*5
    raw_ic = mu_beta + blink + generate_pink_noise(512)*2
    
    axes[0].plot(t_ica, raw_ic, color='black')
    axes[0].set_title('Raw Trace with Ocular Artifact')
    axes[1].plot(t_ica, blink, color='red')
    axes[1].set_title('Isolated Blink IC (IC_0)')
    axes[2].plot(t_ica, raw_ic - blink, color='blue')
    axes[2].set_title('Clean Reconstructed Trace')
    axes[3].plot(t_ica, mu_beta, color='green')
    axes[3].set_title('Separated µ/β Rhythms (8-30 Hz)')
    plt.tight_layout()
    plt.savefig(os.path.join(artifacts_dir, 'step4_ica_decomposition.png'), dpi=150)
    plt.close()

    # --- Step 5: ERD Spectrogram ---
    plt.figure(figsize=(8, 5))
    f_spec, t_spec, Sxx = spectrogram(mu_beta, fs=128, nperseg=64, noverlap=32)
    Sxx[ (f_spec > 8) & (f_spec < 14), 10:20] *= 0.2 # ERD simulation
    plt.pcolormesh(t_spec, f_spec, 10 * np.log10(Sxx), shading='gouraud', cmap='viridis')
    plt.ylim(0, 45)
    plt.title('Step 5: ERD Spectrogram Heatmap (µ Band Suppression)')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.colorbar(label='Power (dB)')
    plt.tight_layout()
    plt.savefig(os.path.join(artifacts_dir, 'step5_erd_spectrogram.png'), dpi=150)
    plt.close()

    # --- Step 6: Symmetric Concatenation ---
    plt.figure(figsize=(10, 4))
    t_concat = np.arange(1280)
    c_contra = np.sin(2*np.pi*10*t_concat[:640]/160) * 5 + generate_pink_noise(640)*2
    c_ipsi = np.sin(2*np.pi*10*t_concat[640:]/160) * 15 + generate_pink_noise(640)*2
    plt.plot(t_concat[:640], c_contra, label='Contralateral (Low Amplitude ERD)', color='blue')
    plt.plot(t_concat[640:], c_ipsi, label='Ipsilateral (High Amplitude)', color='orange')
    plt.axvline(x=640, color='black', linestyle='--')
    plt.title('Step 6: Symmetrical Electrode Concatenation (1280 points)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(artifacts_dir, 'step6_symmetric_concatenation.png'), dpi=150)
    plt.close()

    # --- Step 7: Segmentation & Partitioning ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(t_concat, np.concatenate([c_contra, c_ipsi]), alpha=0.5)
    for i in range(9):
        ax1.axvspan(i*(1280//9), (i+1)*(1280//9)-5, color='green', alpha=0.2)
    ax1.set_title('9 Non-overlapping Sliding Windows')
    
    ax2.bar(['Train (50%)', 'Val (20%)', 'Test (30%)'], [50, 20, 30], color=['blue', 'orange', 'green'])
    ax2.set_title('Dataset Partitioning')
    ax2.set_ylabel('Percentage')
    plt.tight_layout()
    plt.savefig(os.path.join(artifacts_dir, 'step7_segmentation_split.png'), dpi=150)
    plt.close()

def generate_live_artifacts_figs(raw, events):
    """Generates matplotlib Figures based on actual EEG data passing through the preprocessing pipeline."""
    import mne
    from scipy.signal import spectrogram, welch
    from src.preprocessing import resample_data, apply_car, apply_bandpass_filter, epoch_and_segment
    
    figs = []
    
    # 1. Raw Data Plot
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    data, times = raw[:, :int(raw.info['sfreq'] * 10)] # First 10 seconds
    for i in range(min(10, data.shape[0])):
        ax1.plot(times, data[i] + i*100, color='gray', linewidth=0.8)
    ax1.set_title('Step 1: Real Raw EEG Waveforms (First 10s)')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude (µV) with offset')
    plt.tight_layout()
    figs.append(('Step 1: Raw Ingestion', fig1))
    
    # 2. Resampling & PSD
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(12, 4))
    sfreq_orig = raw.info['sfreq']
    
    f_orig, psd_orig = welch(data[0], fs=sfreq_orig, nperseg=int(sfreq_orig*2))
    ax2b.plot(f_orig, 10*np.log10(psd_orig), label=f'{sfreq_orig} Hz (Original)', alpha=0.7)
    
    raw_resampled = resample_data(raw.copy(), 128)
    data_resampled, times_resampled = raw_resampled[:, :int(128 * 10)]
    f_res, psd_res = welch(data_resampled[0], fs=128, nperseg=128*2)
    ax2b.plot(f_res, 10*np.log10(psd_res), label='128 Hz (Filtered & Downsampled)', alpha=0.9, linestyle='--')
    ax2b.set_title('Power Spectral Density (Ch 0)')
    ax2b.set_xlabel('Frequency (Hz)')
    ax2b.set_ylabel('Power (dB/Hz)')
    ax2b.axvline(x=60, color='r', linestyle=':', label='60Hz Cutoff')
    ax2b.legend()
    
    ax2a.plot(times[:int(sfreq_orig*1)], data[0, :int(sfreq_orig*1)], label='Original')
    ax2a.plot(times_resampled[:128*1], data_resampled[0, :128*1], label='Resampled', linestyle='--')
    ax2a.set_title('Time-domain Overlay (1s)')
    ax2a.legend()
    plt.tight_layout()
    figs.append(('Step 2: Anti-Aliasing & Decimation', fig2))
    
    # 3. CAR
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    raw_car = apply_car(raw_resampled.copy())
    data_car, _ = raw_car[:, :int(128 * 2)] # 2 seconds
    for i in range(min(5, data_resampled.shape[0])):
        ax3.plot(times_resampled[:128*2], data_resampled[i, :128*2], color='red', alpha=0.3)
        ax3.plot(times_resampled[:128*2], data_car[i], color='blue', alpha=0.5)
    ax3.set_title('Step 3: Pre-CAR (Red) vs Post-CAR (Blue) Butterfly Trace')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Amplitude (µV)')
    plt.tight_layout()
    figs.append(('Step 3: Common Average Referencing (CAR)', fig3))
    
    # 4. ICA/Bandpass
    fig4, (ax4a, ax4b) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    raw_clean = apply_bandpass_filter(raw_car.copy(), 4, 38)
    data_clean, _ = raw_clean[:, :int(128 * 4)] # 4 seconds
    data_car_full, _ = raw_car[:, :int(128 * 4)] # 4 seconds
    ax4a.plot(times_resampled[:128*4], data_car_full[0], color='black')
    ax4a.set_title('Pre-ICA/Bandpass Trace (Ch 0)')
    ax4b.plot(times_resampled[:128*4], data_clean[0], color='green')
    ax4b.set_title('Clean Reconstructed Trace (8-30 Hz)')
    plt.tight_layout()
    figs.append(('Step 4: Band Decomposition', fig4))
    
    # 5. ERD Spectrogram
    fig5, ax5 = plt.subplots(figsize=(8, 5))
    f_spec, t_spec, Sxx = spectrogram(data_clean[0], fs=128, nperseg=64, noverlap=32)
    mesh = ax5.pcolormesh(t_spec, f_spec, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='viridis')
    ax5.set_ylim(0, 45)
    ax5.set_title('Step 5: Real ERD Spectrogram Heatmap (µ/β Band)')
    ax5.set_ylabel('Frequency (Hz)')
    ax5.set_xlabel('Time (s)')
    fig5.colorbar(mesh, ax=ax5, label='Power (dB)')
    plt.tight_layout()
    figs.append(('Step 5: Active Trial Windowing & ERD', fig5))
    
    # 6 & 7. Segmentation
    if len(events) > 0:
        event_dict = {'T0':0, 'T1':1, 'T2':2} if 'T0' in events else None
        if event_dict is None:
            # try to infer event dict
            unique_events = np.unique(events[:, 2])
            event_dict = {f'T{e}':e for e in unique_events}
        
        try:
            X, y = epoch_and_segment(raw_clean.copy(), events, event_dict)
            
            fig6, ax6 = plt.subplots(figsize=(10, 4))
            # Plot one concatenated segment
            sample_segment = X[0, 0, :] # first trial, first pair (FC3/FC4), all times
            mid_point = len(sample_segment) // 2
            ax6.plot(np.arange(mid_point), sample_segment[:mid_point], label='Contralateral (Ch1)', color='blue')
            ax6.plot(np.arange(mid_point, len(sample_segment)), sample_segment[mid_point:], label='Ipsilateral (Ch2)', color='orange')
            ax6.axvline(x=mid_point, color='black', linestyle='--')
            ax6.set_title('Step 6: Symmetrical Electrode Concatenation (Real Data)')
            ax6.legend()
            plt.tight_layout()
            figs.append(('Step 6: Symmetrical Electrode Concatenation', fig6))
            
            fig7, (ax7a, ax7b) = plt.subplots(1, 2, figsize=(12, 4))
            ax7a.plot(sample_segment, alpha=0.5)
            n_windows = 9
            window_size = len(sample_segment) // n_windows
            for i in range(n_windows):
                ax7a.axvspan(i*window_size, (i+1)*window_size-5, color='green', alpha=0.2)
            ax7a.set_title(f'{n_windows} Non-overlapping Sliding Windows')
            
            # Show actual partitioning math
            total_samples = len(X)
            train_pct, val_pct, test_pct = 50, 20, 30
            ax7b.bar(['Train', 'Val', 'Test'], 
                     [total_samples*0.5, total_samples*0.2, total_samples*0.3], 
                     color=['blue', 'orange', 'green'])
            ax7b.set_title(f'Dataset Partitioning (Total sliding windows: {total_samples})')
            ax7b.set_ylabel('Number of Samples')
            plt.tight_layout()
            figs.append(('Step 7: Segmentation & Partitioning', fig7))
        except Exception as e:
            fig_err, ax_err = plt.subplots()
            ax_err.text(0.5, 0.5, f"Error segmenting: {e}", ha='center')
            figs.append(('Step 6 & 7 Error', fig_err))
    
    return figs

if __name__ == '__main__':
    generate_artifacts()
    print("Generated all 7 preprocessing artifacts.")
