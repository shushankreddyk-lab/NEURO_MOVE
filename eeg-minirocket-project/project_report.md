# Project Report: Advanced EEG-Based Motor Imagery Classification using MiniRocket and CNN-LSTM Architectures

## 1. Abstract
Brain-Computer Interfaces (BCIs) translate neural activity into control commands, providing a direct communication pathway between the human brain and external devices. A critical challenge in Electroencephalography (EEG)-based Motor Imagery (MI) BCIs is accurately and efficiently decoding highly non-stationary and noisy neural signals. This project proposes a robust and scalable MI-BCI framework utilizing the MiniRocket (Mini RandOm Convolutional KErnel Transform) architecture and CNN-LSTM hybrid models. Unlike traditional deep learning approaches that suffer from high computational overhead and prolonged training times, our optimized MiniRocket engine achieves rapid convergence while maintaining high classification accuracy. The system is equipped with an interactive, real-time Live Training Console built in Streamlit, which dynamically parses, visualizes, and trains on diverse multi-channel EEG datasets. The framework was rigorously evaluated on two standard benchmarks: the 109-subject PhysioNet EEGMMIDB dataset (64-channel) and the BCI Competition IV 2a dataset (22-channel). Through the application of Common Average Referencing (CAR) and customized spectral bandpass filtering (4–38 Hz), the proposed system demonstrates substantial improvements in signal-to-noise ratio and classification performance, establishing a highly accessible and extensible ecosystem for real-time BCI applications.

## 2. Introduction
Motor Imagery (MI) involves the mental simulation of a specific motor action without actual execution. This cognitive process triggers Event-Related Desynchronization (ERD) and Event-Related Synchronization (ERS) in the sensorimotor cortex, which can be captured non-invasively via EEG. Recognizing these patterns enables individuals, especially those with severe neuromuscular disorders, to control prosthetics, wheelchairs, or digital interfaces using thought alone.

Despite significant advancements, traditional MI classification methods, such as Common Spatial Pattern (CSP) combined with Support Vector Machines (SVM), often require extensive subject-specific calibration. Conversely, recent Deep Learning models (e.g., Deep ConvNets, EEGNet) offer automatic feature extraction but demand substantial computational resources, limiting their deployment in practical, low-latency, or embedded clinical BCI settings.

To bridge this gap, this project develops a comprehensive BCI ecosystem named *NEURO_MOVE*. The core contribution is the integration of the MiniRocket feature extractor—a state-of-the-art time-series classification algorithm that applies thousands of deterministic, non-dilated convolutional kernels to EEG signals at a fraction of the computational cost of standard CNNs. This is complemented by a CNN-LSTM architecture for capturing complex spatiotemporal dynamics. Furthermore, the project addresses the critical need for researchers to interactively manage and train on massive EEG datasets by introducing a dynamic Live Training Console capable of parsing heterogeneous file formats (.edf and .gdf) on the fly.

## 3. Literature Review
The decoding of MI-EEG signals has traditionally relied on spatial filtering techniques. Blankertz et al. demonstrated the efficacy of Common Spatial Patterns (CSP) for maximizing the variance of EEG signals for one class while minimizing it for another. However, CSP is highly susceptible to noise, artifacts, and non-stationarity across different subjects and sessions.

The advent of deep learning brought a paradigm shift to BCI research. Schirrmeister et al. introduced Shallow and Deep ConvNets specifically designed for raw EEG decoding, demonstrating that CNNs could learn optimal spatial and temporal filters directly from the data. Following this, Lawhern et al. proposed EEGNet, a compact CNN architecture that utilized depthwise and separable convolutions to significantly reduce the parameter count while maintaining robustness across different BCI paradigms.

While CNNs and hybrid models like CNN-LSTM (which leverage Recurrent Neural Networks to model temporal dependencies) yield high accuracy, they often struggle with the "curse of dimensionality" and require extensive training epochs on large GPUs. Dempsey et al. revolutionized time-series classification by introducing ROCKET and its optimized successor, MiniRocket. MiniRocket transforms time-series data using a small fixed set of convolutional kernels, extracting features almost instantly. Recent literature suggests that applying MiniRocket to multichannel EEG can achieve near state-of-the-art accuracy with training times reduced by orders of magnitude compared to traditional CNNs, making it an ideal candidate for scalable, real-time BCI frameworks.

## 4. Proposed Methodology
The proposed *NEURO_MOVE* framework consists of a complete pipeline encompassing data ingestion, preprocessing, feature extraction, and interactive classification.

### 4.1 Data Ingestion and Live Console
An interactive dashboard was developed to manage datasets. The backend utilizes custom binary parsers to dynamically scan directories, read `.edf` (PhysioNet) and `.gdf` (BCI 2a) files using the `mne-python` library, and extract event annotations. A "Live Training Console" generates a detailed Table of Contents (TOC) mapping raw files to specific motor tasks (e.g., Left Hand, Right Hand, Foot, Tongue, and Rest) across all subjects.

### 4.2 Preprocessing and Spatial Filtering
Raw EEG signals are inherently contaminated by environmental noise and ocular/muscular artifacts. The preprocessing pipeline includes:
- **Bandpass Filtering**: Signals are filtered between 4 Hz and 38 Hz to isolate the Mu (8–12 Hz) and Beta (13–30 Hz) bands, which are the primary carriers of ERD/ERS features during motor imagery.
- **Common Average Reference (CAR)**: A spatial filter is applied to re-reference the data by subtracting the mean of all electrodes from each individual electrode at every time point. This significantly enhances the signal-to-noise ratio and mitigates global noise artifacts.
- **Epoch Extraction**: Continuous data is segmented into fixed-length windows (e.g., 0 to 4.6 seconds relative to the event cue) to capture the full physiological response of the imagined movement.

### 4.3 Classification Architectures
Two primary modeling strategies are implemented:
1.  **MiniRocket Engine**: The preprocessed, multi-channel EEG epochs are fed into a MiniRocket transformer. The engine applies 10,000 specific convolutional kernels to the time series, extracting a robust feature space based on the Proportion of Positive Values (PPV). A linear classifier (e.g., Ridge Regression or Logistic Regression) is then rapidly fitted to these features, allowing for near-instantaneous training across large cohorts.
2.  **CNN-LSTM Hybrid**: For deeper temporal feature learning, a hybrid model utilizes 1D-Convolutional layers to extract spatial representations, followed by Long Short-Term Memory (LSTM) layers to decode the sequential evolution of the brainwaves over the task duration. 

## 5. Dataset Description
To ensure the proposed models are robust and generalize well across varying channel configurations and experimental paradigms, the system is validated on two widely recognized public datasets.

### 5.1 PhysioNet EEG Motor Movement/Imagery Dataset (EEGMMIDB)
- **Subjects**: 109 healthy subjects.
- **Channels**: 64 EEG electrodes (international 10-10 system).
- **Sampling Rate**: 160 Hz.
- **Paradigm**: Subjects performed 14 experimental runs comprising baseline (eyes open/closed) and multiple motor imagery/execution tasks (Left/Right Fist, Both Fists, Both Feet).
- **Format**: European Data Format (`.edf`).

### 5.2 BCI Competition IV Dataset 2a
- **Subjects**: 9 subjects.
- **Channels**: 22 EEG electrodes and 3 EOG channels.
- **Sampling Rate**: 250 Hz.
- **Paradigm**: A 4-class motor imagery task involving the imagination of movement of the Left Hand, Right Hand, Both Feet, and Tongue. The dataset provides distinct Training and Evaluation sets.
- **Format**: General Data Format (`.gdf`).

The integration of these diverse datasets within the platform proves the architecture's ability to seamlessly adapt to varying input dimensionalities (64 vs. 22 channels) and multi-class classification challenges.
