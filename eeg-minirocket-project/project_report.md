# Project Report: NEURO_MOVE - Advanced EEG-Based Motor Imagery Classification

## 1. Abstract
Brain-Computer Interfaces (BCIs) translate neural activity into control commands, providing a direct communication pathway between the human brain and external devices. A critical challenge in Electroencephalography (EEG)-based Motor Imagery (MI) BCIs is accurately and efficiently decoding highly non-stationary and noisy neural signals. This project proposes a robust and scalable MI-BCI framework utilizing the MiniRocket (Mini RandOm Convolutional KErnel Transform) architecture and CNN-LSTM hybrid models. Unlike traditional deep learning approaches that suffer from high computational overhead and prolonged training times, our optimized MiniRocket engine achieves rapid convergence while maintaining high classification accuracy. The system is equipped with an interactive, real-time Live Training Console built in Streamlit, which dynamically parses, visualizes, and trains on diverse multi-channel EEG datasets. The framework was rigorously evaluated on two standard benchmarks: the 109-subject PhysioNet EEGMMIDB dataset (64-channel) and the BCI Competition IV 2a dataset (22-channel). Through the application of Common Average Referencing (CAR) and customized spectral bandpass filtering (4–38 Hz), the proposed system demonstrates substantial improvements in signal-to-noise ratio and classification performance, establishing a highly accessible and extensible ecosystem for real-time BCI applications.

## 2. Introduction
Motor Imagery (MI) involves the mental simulation of a specific motor action without actual execution. This cognitive process triggers Event-Related Desynchronization (ERD) and Event-Related Synchronization (ERS) in the sensorimotor cortex, which can be captured non-invasively via EEG. Recognizing these patterns enables individuals, especially those with severe neuromuscular disorders, to control prosthetics, wheelchairs, or digital interfaces using thought alone.

Despite significant advancements, traditional MI classification methods, such as Common Spatial Pattern (CSP) combined with Support Vector Machines (SVM), often require extensive subject-specific calibration. Conversely, recent Deep Learning models (e.g., Deep ConvNets, EEGNet) offer automatic feature extraction but demand substantial computational resources, limiting their deployment in practical, low-latency, or embedded clinical BCI settings.

To bridge this gap, this project develops a comprehensive BCI ecosystem named *NEURO_MOVE*. The core contribution is the integration of the MiniRocket feature extractor—a state-of-the-art time-series classification algorithm that applies thousands of deterministic, non-dilated convolutional kernels to EEG signals at a fraction of the computational cost of standard CNNs.

## 3. Literature Review
The decoding of MI-EEG signals has traditionally relied on spatial filtering and classical machine learning techniques. For example, Sita and Nair (2013) achieved an average accuracy of 87.24% on a 3-class motor imagery task using the PhysioNet dataset by combining Independent Component Analysis (ICA) with Gaussian weighting for feature extraction, followed by Linear Discriminant Analysis (LDA) for classification. Blankertz et al. demonstrated the efficacy of Common Spatial Patterns (CSP) for maximizing the variance of EEG signals for one class while minimizing it for another. However, these classical methods often require heavy manual feature engineering and are highly susceptible to non-stationarity across different subjects and sessions.

The advent of deep learning brought a paradigm shift to BCI research. Schirrmeister et al. introduced Shallow and Deep ConvNets specifically designed for raw EEG decoding, demonstrating that CNNs could learn optimal spatial and temporal filters directly from the data. Following this, Lawhern et al. proposed EEGNet, a compact CNN architecture that utilized depthwise and separable convolutions to significantly reduce the parameter count while maintaining robustness across different BCI paradigms.

More recently, Transformer models have emerged as powerful tools for EEG analysis. By utilizing Self-Attention mechanisms, Transformers can capture complex, global temporal and spatial dependencies across the entire EEG epoch without relying on localized convolutional windows. While Transformers offer high accuracy, they suffer from extreme computational costs and require massive datasets to avoid overfitting, making them challenging to deploy in low-latency clinical BCIs.

To balance these extremes, this project leverages the MiniRocket architecture. Recent literature suggests that applying MiniRocket to multichannel EEG can achieve near state-of-the-art accuracy with training times reduced by orders of magnitude compared to traditional CNNs and Transformers, making it an ideal candidate for scalable, real-time BCI frameworks.

## 4. Project Evolution & Lifecycle (From Inception to Roadmap)
The *NEURO_MOVE* project was conceptualized to be more than just a static machine learning script; it is designed as a fully interactive, end-to-end platform for EEG researchers.

- **Phase 1: Foundation (Where We Started)**: We began by implementing a raw pipeline using the PhysioNet EEGMMIDB dataset, utilizing standard `.edf` parsing. The initial focus was on overcoming the massive computational bottleneck of training on 109 subjects. By integrating the MiniRocket architecture, we reduced training times from hours to mere seconds, proving the viability of fast, deterministic kernel transformations on raw EEG.
- **Phase 2: Platform Architecture & UI Development**: To make the tool accessible, we wrapped the backend Python training engine in a sophisticated Streamlit Dashboard (`app.py`). This introduced a "Live Training Console", real-time training telemetry, dynamic dataset scanning (TOC generation), and model architecture visualization.
- **Phase 3: Dataset Expansion & Preprocessing Hardening**: We expanded the platform's capability to ingest the BCI Competition IV 2a dataset (shifting from 64-channel `.edf` to 22-channel `.gdf` files). This required overhauling the spatial filtering (adding Common Average Referencing - CAR), fixing temporal epochs (extending targets to 4.6 seconds to capture ERS), and resolving severe class imbalances.
- **Phase 4: Multi-Model Scaling (How We Are Going to End)**: The final phase involves extending the dashboard to become a comprehensive benchmarking suite. We are currently actively expanding the platform beyond MiniRocket and CNN-LSTM to systematically include state-of-the-art architectures such as EEGNet, Shallow/Deep ConvNets, and classical CSP+LDA combinations. The ultimate goal is a unified platform where a user can upload raw EEG, select any modern architecture, and instantly compare decoding performance in real-time.

## 5. Proposed Methodology
The proposed *NEURO_MOVE* framework consists of a complete pipeline encompassing data ingestion, preprocessing, feature extraction, and interactive classification.

### 5.1 Data Ingestion and Live Console
An interactive dashboard was developed to manage datasets. The backend utilizes custom binary parsers to dynamically scan directories, read `.edf` (PhysioNet) and `.gdf` (BCI 2a) files using the `mne-python` library, and extract event annotations. A "Live Training Console" generates a detailed Table of Contents (TOC) mapping raw files to specific motor tasks across all subjects.

### 5.2 Preprocessing and Spatial Filtering
Raw EEG signals are inherently contaminated by environmental noise and ocular/muscular artifacts. The preprocessing pipeline includes:
- **Bandpass Filtering**: Signals are filtered between 4 Hz and 38 Hz to isolate the Mu (8–12 Hz) and Beta (13–30 Hz) bands, which are the primary carriers of ERD/ERS features during motor imagery.
- **Common Average Reference (CAR)**: A spatial filter is applied to re-reference the data by subtracting the mean of all electrodes from each individual electrode at every time point. This significantly enhances the signal-to-noise ratio and mitigates global noise artifacts.
- **Epoch Extraction**: Continuous data is segmented into fixed-length windows (e.g., 0 to 4.6 seconds relative to the event cue) to capture the full physiological response of the imagined movement.

### 5.3 Classification Architectures
Two primary modeling strategies are currently active:
1.  **MiniRocket Engine**: The preprocessed, multi-channel EEG epochs are fed into a MiniRocket transformer. The engine applies 10,000 specific convolutional kernels to the time series, extracting a robust feature space based on the Proportion of Positive Values (PPV). A linear classifier (e.g., Ridge Regression or Logistic Regression) is then rapidly fitted to these features, allowing for near-instantaneous training across large cohorts.
2.  **CNN-LSTM Hybrid**: For deeper temporal feature learning, a hybrid model utilizes 1D-Convolutional layers to extract spatial representations, followed by Long Short-Term Memory (LSTM) layers to decode the sequential evolution of the brainwaves over the task duration. 

## 6. Dataset Description
Our framework dynamically adapts to the specific data shapes, sampling rates, and labels of each dataset.

### 6.1 PhysioNet EEG Motor Movement/Imagery Dataset (EEGMMIDB)
- **Demographics & Cohort**: 109 healthy volunteer subjects.
- **Hardware & Sensor Setup**: Data was recorded using the BCI2000 system equipped with 64 EEG electrodes (10-10 system). 
- **Sampling Rate**: Digitized at 160 Hz.
- **Protocol**: 14 experimental runs generating over 1,500 total `.edf` files. Includes Baseline (eyes open/closed) and Task Runs (Motor Execution/Imagery for Unilateral and Bilateral fists/feet).
- **Integration**: The Live Training Console processes and extracts tens of thousands of individual epochs from this dataset for high-throughput batch training.

### 6.2 BCI Competition IV Dataset 2a (GDF)
- **Demographics & Cohort**: 9 subjects.
- **Hardware & Sensor Setup**: Recorded using 22 Ag/AgCl EEG electrodes and 3 monopolar EOG channels.
- **Sampling Rate**: Sampled at 250 Hz, bandpass-filtered between 0.5 Hz and 100 Hz, with a 50 Hz notch filter.
- **Protocol**: 4-class motor imagery (Left Hand, Right Hand, Both Feet, Tongue). Split into Training and Evaluation sessions on different days (6 runs of 48 trials each per session).
- **Integration**: The `dataset_2a_loader` script specifically targets the critical 4.6-second task window, applying CAR and 4–38 Hz filtering. 

## 7. Current Results and Platform Status
To date, the `NEURO_MOVE` platform demonstrates exceptionally fast ingestion and training capabilities. 
- On the PhysioNet dataset, the MiniRocket engine successfully trains on tens of thousands of epochs across multiple subjects in under 30 seconds, significantly outperforming traditional LSTM pipelines in training speed.
- On the highly challenging BCI Competition IV 2a dataset, the integration of CAR and targeted epoch extraction has improved base validation accuracies on the 4-class problem from random chance (25%) up to 43.3% using MiniRocket and 39.4% using CNN-LSTM, establishing a solid baseline for a notoriously difficult dataset.

## 8. Future Roadmap
The immediate next steps for the completion of the `NEURO_MOVE` ecosystem involve scaling the comparative capabilities of the platform:
1. **Integration of EEGNet**: Implementing the compact CNN architecture to leverage Depthwise and Separable Convolutions specifically tailored for EEG.
2. **Integration of Shallow and Deep ConvNets**: Adapting Schirrmeister’s proven raw-EEG decoders to benchmark against MiniRocket’s speed.
3. **Integration of CSP + LDA**: Adding the classical machine learning baseline to allow researchers to directly compare deep learning methods against traditional spatial filtering in the Live Training Console.

## 9. Conclusion
The *NEURO_MOVE* framework successfully addresses the computational bottlenecks inherent in modern BCI research. By combining an intuitive, dynamic Streamlit dashboard with blazing-fast feature extractors like MiniRocket and robust pipelines that support diverse, complex datasets, this project provides a powerful, scalable ecosystem. As the platform expands to include EEGNet and CSP, it will serve as a definitive, all-in-one benchmarking suite for Motor Imagery Brain-Computer Interfaces.

