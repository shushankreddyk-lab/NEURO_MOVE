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
To ensure the proposed MiniRocket and CNN-LSTM models are robust and generalize well across varying channel configurations and experimental paradigms, the system is validated on two widely recognized public datasets. Our framework dynamically adapts to the specific data shapes, sampling rates, and labels of each dataset.

### 5.1 PhysioNet EEG Motor Movement/Imagery Dataset (EEGMMIDB)
The PhysioNet dataset is one of the largest publicly available EEG databases for motor imagery, designed to capture complex, multi-task neural responses across a large cohort.

- **Demographics & Cohort**: 109 healthy volunteer subjects.
- **Hardware & Sensor Setup**: Data was recorded using the BCI2000 system equipped with 64 EEG electrodes distributed according to the international 10-10 system. 
- **Sampling Rate & Resolution**: Signals were digitized at 160 Hz.
- **Experimental Protocol**: Each subject completed 14 separate experimental runs, generating over 1,500 total `.edf` (European Data Format) files across the cohort. The sessions were divided into:
  - **Baseline Runs (R01, R02)**: One minute of eyes-open and one minute of eyes-closed resting state.
  - **Task Runs**: Four distinct task sets performed three times each (R03-R14). Subjects were presented with a target on a screen and prompted to perform or imagine a motor task.
- **Task Classifications**:
  1. *Motor Execution (Unilateral)*: Opening and closing the left or right fist.
  2. *Motor Imagery (Unilateral)*: Imagining opening and closing the left or right fist.
  3. *Motor Execution (Bilateral)*: Opening and closing both fists or both feet.
  4. *Motor Imagery (Bilateral)*: Imagining opening and closing both fists or both feet.
- **Integration in NEURO_MOVE**: Our custom binary parser extracts `T0` (Rest), `T1`, and `T2` annotations dynamically. The Live Training Console successfully processes and extracts tens of thousands of individual epochs from this dataset for high-throughput batch training.

### 5.2 BCI Competition IV Dataset 2a (GDF)
The BCI Competition IV 2a dataset is considered a gold standard for evaluating multi-class motor imagery algorithms, posing a significant challenge due to its highly non-stationary nature and limited subject count.

- **Demographics & Cohort**: 9 subjects.
- **Hardware & Sensor Setup**: Recorded using 22 Ag/AgCl EEG electrodes (sampling the primary motor and sensorimotor cortices) and 3 monopolar EOG channels (to track and filter ocular artifacts).
- **Sampling Rate & Resolution**: Data was sampled at 250 Hz and bandpass-filtered natively between 0.5 Hz and 100 Hz, with a 50 Hz notch filter applied to suppress power line noise.
- **Experimental Protocol**: The dataset is explicitly split into two sessions per subject (Training and Evaluation), recorded on different days to test the temporal generalization of algorithms. Each session consisted of 6 runs with 48 trials each (yielding 288 trials per session).
- **Task Classifications (4-Class MI)**:
  - Left Hand (Marker `769`)
  - Right Hand (Marker `770`)
  - Both Feet (Marker `771`)
  - Tongue (Marker `772`)
- **Trial Structure**: A typical trial begins with a fixation cross and an acoustic warning (t = 0 s). A visual cue indicating the specific motor imagery task (left, right, foot, tongue) is presented from t = 2.0 s to t = 3.25 s. Subjects perform the imagined movement until t = 6.0 s.
- **Integration in NEURO_MOVE**: Our custom `dataset_2a_loader` script specifically targets the critical task window (extending to 4.6 seconds) and employs Common Average Referencing (CAR) alongside 4–38 Hz bandpass filtering to extract powerful ERD/ERS signatures. The framework parses the `.gdf` formats seamlessly, successfully increasing baseline accuracy significantly over random chance using the MiniRocket engine.
