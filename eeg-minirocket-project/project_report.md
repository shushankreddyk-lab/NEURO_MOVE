# Project Report: NEURO_MOVE - Advanced EEG-Based Motor Imagery Classification Ecosystem

## 1. Abstract
Brain-Computer Interfaces (BCIs) represent a transformative frontier in human-computer interaction, offering a direct communication pathway between the human brain and external devices by translating neural activity into executable control commands. Among the various non-invasive modalities, Electroencephalography (EEG)-based Motor Imagery (MI) is particularly prominent due to its high temporal resolution, portability, and non-reliance on external stimuli. However, a critical challenge in MI-BCIs is the accurate and efficient decoding of highly non-stationary, low signal-to-noise ratio (SNR) neural signals across diverse subjects. Traditional approaches often rely on handcrafted spatial filters, while modern deep learning methods suffer from extreme computational overhead, impeding their use in real-time, low-latency environments. 

This project, titled *NEURO_MOVE*, proposes a highly robust, scalable, and interactive MI-BCI framework. The core innovation lies in the deployment of the MiniRocket (Mini RandOm Convolutional KErnel Transform) architecture alongside CNN-LSTM hybrid models. Unlike conventional deep neural networks that require prolonged training epochs on GPU clusters, our optimized MiniRocket engine achieves extremely rapid convergence by applying thousands of deterministic kernels, maintaining state-of-the-art classification accuracy while reducing training times to mere seconds. The entire framework is embedded within an interactive, real-time Live Training Console constructed in Streamlit. This dashboard dynamically parses raw data files, visualizes class distributions, and executes batch training. 

The system’s robustness and generalizability were rigorously evaluated across two standard benchmarks: the massive 109-subject PhysioNet EEG Motor Movement/Imagery Dataset (comprising 64-channel recordings) and the notoriously difficult BCI Competition IV 2a dataset (22-channel recordings). Through the strategic application of Common Average Referencing (CAR) and customized spectral bandpass filtering isolating the Mu (8–12 Hz) and Beta (13–30 Hz) bands, the proposed architecture demonstrates substantial improvements in signal fidelity and classification performance. *NEURO_MOVE* establishes a highly accessible, extensible ecosystem for rapid prototyping and deployment of real-time BCI applications.

## 2. Introduction
### 2.1 Background and Motivation
Motor Imagery (MI) involves the conscious, mental simulation of a specific motor action (such as moving a hand or foot) without any actual muscular execution. Neurophysiologically, this cognitive process triggers localized power decreases (Event-Related Desynchronization, ERD) and subsequent power increases (Event-Related Synchronization, ERS) in the primary motor and sensorimotor cortices. These phenomena can be captured non-invasively via Electroencephalography (EEG). Recognizing and decoding these intricate spatial-temporal patterns empowers individuals—especially those suffering from severe neuromuscular disorders such as Amyotrophic Lateral Sclerosis (ALS) or spinal cord injuries—to operate neuroprosthetics, motorized wheelchairs, or digital communication interfaces using solely their thoughts.

### 2.2 Challenges in MI-EEG Decoding
Despite significant academic and clinical interest, translating MI-BCI systems from controlled laboratory environments to practical, real-world applications remains fraught with challenges:
1.  **Low Signal-to-Noise Ratio (SNR):** EEG recordings are heavily contaminated by environmental electrical noise (e.g., 50/60 Hz power line interference) and physiological artifacts (e.g., electrooculography (EOG) from eye blinks, and electromyography (EMG) from muscle tension).
2.  **High Dimensionality and Non-Stationarity:** EEG data is highly dimensional (multi-channel, high sampling rate) and notoriously non-stationary. The neural responses vary drastically not only between different subjects (inter-subject variability) but also within the same subject across different recording sessions (intra-subject variability).
3.  **Computational Bottlenecks:** Modern Deep Learning architectures designed to handle this complexity demand substantial computational resources, rendering them difficult to deploy in embedded systems or portable BCI hardware that require near-instantaneous feedback.

### 2.3 Objectives and Contributions
To bridge the gap between high accuracy and computational efficiency, this project develops the *NEURO_MOVE* ecosystem. The primary objectives and contributions include:
- The integration of the **MiniRocket** feature extractor, a state-of-the-art time-series classification algorithm that bypasses traditional gradient-descent optimization for feature extraction, drastically reducing training times.
- The development of a **CNN-LSTM** hybrid model to decode deeper sequential evolution and temporal dependencies in brainwaves.
- The creation of an end-to-end **Streamlit-based UI**, providing researchers with a "Live Training Console" that completely abstracts the complexities of data parsing, preprocessing, and model evaluation.

## 3. Literature Review
The decoding of MI-EEG signals has experienced a significant evolution over the past two decades, transitioning from classical spatial filtering techniques to complex deep learning architectures.

### 3.1 Classical Machine Learning and Spatial Filtering
Historically, BCI systems have relied heavily on spatial filtering and classical machine learning. The most prominent technique is the **Common Spatial Pattern (CSP)** algorithm. Blankertz et al. demonstrated the efficacy of CSP in maximizing the variance of spatially filtered EEG signals for one class (e.g., left hand movement) while simultaneously minimizing the variance for another (e.g., right hand movement). 

Building upon these classical foundations, researchers sought to optimize feature extraction. For instance, **Sita and Nair (2013)** demonstrated a highly effective approach by combining Independent Component Analysis (ICA) with Gaussian weighting. ICA was utilized to separate the mixed EEG signals into statistically independent subcomponents, effectively isolating the specific brain waves related to motor imagery from background noise and ocular artifacts. Following this extraction, they employed Linear Discriminant Analysis (LDA) and Fast Discriminant Analysis (FDA) for classification. Evaluating their framework on the PhysioNet dataset, Sita and Nair achieved an impressive average accuracy of 87.24% on a 3-class problem (Right Fist, Left Fist, and Both Feet). However, while such classical methods can yield high accuracy, they often require heavy, manual feature engineering, precise hyperparameter tuning, and are highly susceptible to the non-stationarity of brainwaves across different subjects and sessions.

### 3.2 Deep Learning Architectures
The advent of deep learning brought a paradigm shift, eliminating the need for manual feature engineering. 
- **ConvNets:** Schirrmeister et al. introduced the Shallow and Deep ConvNets specifically designed for raw EEG decoding. They demonstrated that Convolutional Neural Networks (CNNs) could automatically learn optimal spatial and temporal filters directly from raw data, bypassing algorithms like CSP entirely.
- **EEGNet:** Following this, Lawhern et al. proposed EEGNet, a highly compact CNN architecture. By utilizing depthwise and separable convolutions, EEGNet significantly reduced the parameter count (making it less prone to overfitting on small BCI datasets) while maintaining robustness across various BCI paradigms including P300, ERN, and MRCP.
- **Transformers:** More recently, Transformer models have emerged as powerful tools for EEG analysis. By utilizing Self-Attention mechanisms, Transformers can capture complex, global temporal and spatial dependencies across the entire EEG epoch without relying on localized convolutional windows. While Transformers offer theoretical advantages in modeling long-range dependencies, they suffer from extreme computational costs and require massive datasets (often millions of trials) to avoid overfitting, making them highly challenging to deploy in standard, low-latency clinical BCIs.

### 3.3 Time-Series Transformers: ROCKET and MiniRocket
While CNNs and Transformers yield high accuracy, they often struggle with the "curse of dimensionality" and require extensive training epochs on large GPUs. Dempsey et al. revolutionized the field of time-series classification by introducing ROCKET (RandOm Convolutional KErnel Transform) and its optimized successor, **MiniRocket**. By generating thousands of deterministic, non-dilated convolutional kernels and extracting the Proportion of Positive Values (PPV), MiniRocket achieves near state-of-the-art accuracy across a vast array of time-series benchmarks while reducing training times by orders of magnitude compared to traditional CNNs. Its application to multichannel EEG represents a critical advancement for scalable, real-time BCI frameworks.

## 4. Project Evolution & Lifecycle (From Inception to Roadmap)
The *NEURO_MOVE* project was conceptualized not as a static script, but as a fully interactive, evolving, end-to-end platform for EEG researchers. The project's lifecycle is divided into four distinct phases:

### Phase 1: Foundation (Where We Started)
The project began by implementing a raw data ingestion pipeline targeting the PhysioNet EEGMMIDB dataset. Utilizing standard `.edf` parsing via the `mne-python` library, the initial focus was on overcoming the massive computational bottleneck of training on 109 subjects (over 1,500 total files). Early attempts using standard deep learning models resulted in excessive training times. By integrating the MiniRocket architecture, we successfully reduced training times from hours to mere seconds, proving the viability of fast, deterministic kernel transformations on raw, multi-channel EEG.

### Phase 2: Platform Architecture & UI Development
To democratize access to the tool, the backend Python training engine was wrapped in a sophisticated, web-based Streamlit Dashboard (`app.py`). This phase introduced the "Live Training Console," providing real-time training telemetry, dynamic dataset scanning, automated Table of Contents (TOC) generation, and interactive model architecture visualizations.

### Phase 3: Dataset Expansion & Preprocessing Hardening
The platform's capabilities were significantly expanded to ingest the BCI Competition IV 2a dataset, representing a shift from 64-channel `.edf` files to 22-channel `.gdf` files. This transition exposed significant challenges with noise and trial alignment. We overhauled the preprocessing pipeline by introducing robust spatial filtering (Common Average Referencing - CAR), fixing temporal epochs (extending targets to 4.6 seconds to explicitly capture ERS phenomena), and resolving severe class imbalances via dynamic trial aggregation.

### Phase 4: Multi-Model Scaling (How We Are Going to End)
The final, ongoing phase of the project involves extending the dashboard into a comprehensive benchmarking suite. We are actively expanding the platform beyond MiniRocket and CNN-LSTM to systematically include state-of-the-art architectures such as EEGNet, Shallow/Deep ConvNets, and classical CSP+LDA combinations. The ultimate roadmap envisions a unified platform where a user can seamlessly upload raw EEG, select any modern architecture, and instantly compare decoding performance and training latency in real-time.

## 5. Proposed Methodology
The proposed *NEURO_MOVE* framework consists of a highly modular pipeline encompassing data ingestion, robust preprocessing, advanced feature extraction, and interactive classification.

### 5.1 Data Ingestion and the Live Console
An interactive dashboard serves as the central command hub. The backend utilizes custom binary parsers to dynamically scan nested directories, reading both `.edf` (PhysioNet) and `.gdf` (BCI 2a) file formats. A specialized parser extracts raw event annotations (e.g., markers `769`, `770`, `771`, `772` for BCI 2a) and maps them to human-readable motor tasks. The Live Training Console generates a detailed Table of Contents (TOC), establishing a cached mapping of raw files to specific motor trials across all subjects, ensuring that the system only loads necessary segments into RAM during training.

### 5.2 Preprocessing and Spatial Filtering
Raw EEG signals are inherently contaminated. The preprocessing pipeline is engineered to maximize the SNR of the ERD/ERS phenomena:
- **Bandpass Filtering:** Signals are strictly filtered between 4 Hz and 38 Hz using Finite Impulse Response (FIR) or infinite impulse response (IIR) Butterworth filters. This explicitly isolates the Mu (8–12 Hz) and Beta (13–30 Hz) bands, which are the primary physiological carriers of motor imagery signatures.
- **Common Average Reference (CAR):** A crucial spatial filter is applied to re-reference the data. CAR subtracts the mean of all electrodes from each individual electrode at every time point. This technique effectively removes widespread, common-mode noise (such as far-field power line interference or generalized biological noise), significantly sharpening the localized focal activity over the motor cortex.
- **Epoch Extraction:** Continuous data streams are segmented into fixed-length windows relative to the onset of the task cue. For example, BCI 2a trials are extracted from 0 to 4.6 seconds, ensuring the entire physiological cascade of the imagined movement is captured for the classifiers.

### 5.3 Classification Architectures
Two primary modeling strategies currently drive the inference engine:

1.  **The MiniRocket Engine:** The preprocessed, multi-channel EEG epochs are flattened and fed into the MiniRocket transformer. The engine applies 10,000 highly specific, non-dilated convolutional kernels to the time series. Instead of generating massive activation maps, MiniRocket simply calculates the Proportion of Positive Values (PPV) for each kernel, resulting in a strictly defined, robust 10,000-dimensional feature space. A linear classifier (such as Ridge Regression Classifier or Logistic Regression) is then rapidly fitted to these features using highly optimized C-level solvers, allowing for near-instantaneous training across massive cohorts.
2.  **CNN-LSTM Hybrid:** For applications requiring deeper temporal feature learning, a hybrid neural network is employed. 1D-Convolutional layers slide across the temporal axis of the EEG channels to extract local spatial-temporal representations. These feature maps are subsequently fed into Long Short-Term Memory (LSTM) recurrent layers, which decode the sequential, long-range evolution of the brainwaves over the 4.6-second task duration.

## 6. Dataset Description
A core strength of the *NEURO_MOVE* architecture is its ability to dynamically adapt to varying data shapes, sampling rates, and label taxonomies without manual code intervention.

### 6.1 PhysioNet EEG Motor Movement/Imagery Dataset (EEGMMIDB)
The PhysioNet dataset is one of the largest publicly available EEG databases for motor imagery, designed to capture complex, multi-task neural responses across a large cohort.

- **Demographics & Cohort:** 109 healthy volunteer subjects.
- **Hardware & Sensor Setup:** Data was recorded using the BCI2000 system equipped with 64 EEG electrodes distributed according to the international 10-10 system. 
- **Sampling Rate & Resolution:** Signals were digitized at 160 Hz.
- **Experimental Protocol:** Each subject completed 14 separate experimental runs, generating over 1,500 total `.edf` (European Data Format) files. The sessions were divided into:
  - *Baseline Runs (R01, R02):* One minute of eyes-open and one minute of eyes-closed resting state.
  - *Task Runs:* Four distinct task sets performed three times each (R03-R14), prompting subjects to perform or imagine a motor task.
- **Task Classifications:**
  1. Motor Execution (Unilateral): Opening/closing left or right fist.
  2. Motor Imagery (Unilateral): Imagining opening/closing left or right fist.
  3. Motor Execution (Bilateral): Opening/closing both fists or both feet.
  4. Motor Imagery (Bilateral): Imagining opening/closing both fists or both feet.
- **Integration:** The system dynamically extracts `T0` (Rest), `T1`, and `T2` annotations, parsing tens of thousands of epochs for high-throughput batch training.

### 6.2 BCI Competition IV Dataset 2a (GDF)
The BCI Competition IV 2a dataset is considered a gold standard for evaluating multi-class motor imagery algorithms, posing a significant challenge due to its highly non-stationary nature and limited subject count.

- **Demographics & Cohort:** 9 subjects.
- **Hardware & Sensor Setup:** Recorded using 22 Ag/AgCl EEG electrodes (sampling the primary motor and sensorimotor cortices) and 3 monopolar EOG channels (to track and filter ocular artifacts).
- **Sampling Rate & Resolution:** Data was sampled at 250 Hz and bandpass-filtered natively between 0.5 Hz and 100 Hz, with a 50 Hz notch filter applied to suppress power line noise.
- **Experimental Protocol:** The dataset is explicitly split into two sessions per subject (Training and Evaluation), recorded on different days to test the temporal generalization of algorithms. Each session consisted of 6 runs with 48 trials each (yielding 288 trials per session).
- **Task Classifications (4-Class MI):** Left Hand (Marker `769`), Right Hand (Marker `770`), Both Feet (Marker `771`), Tongue (Marker `772`).
- **Trial Structure:** A typical trial begins with a fixation cross and an acoustic warning (t = 0 s). A visual cue indicating the specific motor imagery task is presented from t = 2.0 s to t = 3.25 s. Subjects perform the imagined movement until t = 6.0 s.
- **Integration:** The `dataset_2a_loader` script specifically targets the critical task window (extending to 4.6 seconds) and employs Common Average Referencing (CAR) alongside 4–38 Hz bandpass filtering. 

## 7. Current Results and Platform Status
To date, the *NEURO_MOVE* platform demonstrates exceptionally fast ingestion and training capabilities, establishing a robust baseline for ongoing experiments.

- **PhysioNet Scalability:** On the massive PhysioNet dataset, the MiniRocket engine successfully transforms and trains on tens of thousands of epochs across multiple subjects in under 30 seconds. This significantly outperforms traditional LSTM pipelines in training speed, validating the platform's capability to handle large-cohort clinical data in real-time.
- **BCI 2a Performance (4-Class MI):** On the highly challenging BCI Competition IV 2a dataset, the 4-class problem (Left Hand, Right Hand, Foot, Tongue) is notoriously difficult due to extreme inter-session variability. The integration of Common Average Referencing (CAR) and targeted 4.6-second epoch extraction has improved base cross-validation accuracies from a theoretical random chance (25%) up to 43.3% using the MiniRocket engine, and 39.4% using the baseline CNN-LSTM. These results establish a highly solid, reproducible baseline upon which specialized spatial filters can be built.

## 8. Future Roadmap and Next Steps
The immediate next steps for the completion of the *NEURO_MOVE* ecosystem involve scaling the comparative capabilities of the platform to create an ultimate, all-in-one benchmarking suite. The following architectures are slated for immediate integration:

1.  **Integration of EEGNet:** Implementing the compact CNN architecture to leverage Depthwise and Separable Convolutions specifically tailored for EEG. This will allow the platform to evaluate how highly parameterized spatial filters compare to MiniRocket's deterministic kernels.
2.  **Integration of Shallow and Deep ConvNets:** Adapting Schirrmeister’s proven raw-EEG decoders to benchmark state-of-the-art deep learning accuracy against MiniRocket’s computational speed.
3.  **Integration of CSP + LDA / SVM:** Adding the classical machine learning baselines (mirroring the foundational works like Sita and Nair, 2013). This will allow researchers to directly compare deep learning methods against traditional spatial filtering within the same Live Training Console, ensuring a completely fair, apples-to-apples evaluation across datasets.

## 9. Conclusion
The *NEURO_MOVE* framework successfully addresses the critical computational bottlenecks inherent in modern BCI research. By combining an intuitive, dynamic Streamlit dashboard with blazing-fast feature extractors like MiniRocket and robust signal processing pipelines that support diverse, complex datasets, this project provides a powerful, scalable ecosystem. It completely abstracts the difficulty of file parsing, artifact filtering, and hyperparameter tuning. As the platform expands in the coming phases to include EEGNet, Deep ConvNets, and classical CSP configurations, it will serve as a definitive, all-in-one benchmarking suite for Motor Imagery Brain-Computer Interfaces, accelerating the pathway from academic research to real-world clinical deployment.
