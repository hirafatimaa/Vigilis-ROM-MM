# Confidence-Calibrated Adaptive Gated Late Fusion (CC-AGLF) & Vigilis-ROM-MM Benchmark

This repository contains the dataset manifest, error analysis, baseline code, and hydration scripts for the research paper:
**"A Fusion-Based Multimodal Framework for Harmful Content Detection in Code-Mixed Roman Urdu Social Media"**

---

## 📂 Repository Contents

*   **`dataset_manifest.csv`**: The index file for the 50,000-sample Vigilis-ROM-MM benchmark corpus. To comply with platform Terms of Service and GDPR rules, it contains post IDs, platform sources, verified annotations, and split partitions, but no raw media files. Every text entry is 100% unique.
*   **`error_analysis.csv`**: Detailed log containing the classification errors mapped during baseline evaluations, categorized by failure types.
*   **`hydrate_dataset.py`**: Python utility script to download and hydrate public media files (text, audio, and visual tracks) locally on your machine.
*   **`run_ml_experiment.py`**: Main ML pipeline script to train and validate baseline models, ablated configurations, and the proposed CC-AGLF layer.
*   **`README.md`**: This guide.

---

## 🧠 Core Architecture: Confidence-Calibrated Adaptive Gated Late Fusion (CC-AGLF)

To address the challenges of missing modalities and confidence overestimation (poor calibration) in multi-platform social media streams, we propose the **Confidence-Calibrated Adaptive Gated Late Fusion (CC-AGLF)** framework. 

Rather than projecting high-dimensional heterogeneous embeddings (768-d text/visual and 512-d audio) into a shared space—which fails when media channels are missing—CC-AGLF fuses decision-level probability estimators with binary modality-availability indicators.

### 1. Mathematical Formulation

Let $p_t, p_a, p_v \in [0, 1]$ represent the posterior probability outputs of the harmful class from the text, audio, and visual unimodal classifiers, respectively. Let $m_a, m_v \in \{0, 1\}$ represent binary availability indicators where $m_c = 1$ if modality $c$ is present and $0$ otherwise (text is assumed always present, $m_t = 1$).

The adaptive gated feature vector $\mathbf{h} \in \mathbb{R}^5$ is constructed as:

$$\mathbf{h} = \begin{bmatrix} p_t \\ m_a \cdot p_a + (1 - m_a) \cdot 0.5 \\ m_v \cdot p_v + (1 - m_v) \cdot 0.5 \\ m_a \\ m_v \end{bmatrix}$$

Here, missing modalities ($m_a = 0$ or $m_v = 0$) default to a neutral prior probability of $0.5$ to prevent model bias on incomplete inputs.

The final fusion prediction probability is computed via a parameterized sigmoid layer:

$$P(y=1 \mid \mathbf{h}) = \sigma(\mathbf{w}^T \mathbf{h} + b)$$

where $\mathbf{w} \in \mathbb{R}^5$ is the weight vector and $b \in \mathbb{R}$ is the bias parameter, both optimized on the validation set.

### 2. Post-Hoc Calibration

To ensure that predicted probabilities match actual classification frequencies, post-hoc **Platt scaling** is applied to the logistic inputs. For a given logit $z = \mathbf{w}^T \mathbf{h} + b$, the calibrated probability is:

$$\hat{p} = \frac{1}{1 + \exp(A \cdot z + B)}$$

where parameters $A$ and $B$ are fitted using maximum likelihood estimation on the validation set.

---

## 📊 Dataset Curation & Verification

The **Vigilis-ROM-MM** dataset comprises **50,000 fully supervised, manually verified samples** collected from Facebook, X (Twitter), TikTok, YouTube, and Instagram between January 2024 and December 2024.

### 1. Platforms and Modality Availability Counts

| Platform | Count | Modality Text | Modality Audio | Modality Visual |
|---|---|---|---|---|
| Facebook | 12,538 | 12,538 | 10,030 | 12,538 |
| X (Twitter) | 12,548 | 12,548 | 4,391 | 4,391 |
| TikTok | 10,077 | 10,077 | 10,077 | 10,077 |
| YouTube | 7,482 | 7,482 | 7,482 | 7,482 |
| Instagram | 7,355 | 7,355 | 7,355 | 7,355 |
| **Total** | **50,000** | **50,000** | **39,335** | **41,843** |

*   **Class Distribution**: Harmful Class (1): 17,496 samples ($35.0\%$) | Safe Class (0): 32,504 samples ($65.0\%$).
*   **Annotation Quality**: Three native speakers annotated the corpus (Fleiss' Kappa $\kappa = 0.812$). A fourth senior moderator resolved tie-breakers.
*   **IRB Approval & Ethics**: Approved by the CS Department Institutional Review Board of DHA Suffa University under approval number **#IRB-CS-2024-032**. All PII has been stripped, and user IDs hashed using SHA-256.

---

## 📈 Baseline Experimental Results ($N_{test} = 10,000$)

All evaluations are reported exclusively on the held-out test split under a stratified 60/20/20 partition (30,000 train, 10,000 validation, and 10,000 test samples) to ensure 0% information leakage.

| Model Configuration | Accuracy | Precision | Recall | Specificity | F1-Score | ROC-AUC | MCC | ECE | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Proposed CC-AGLF** | **0.9590** | **0.9472** | **0.9351** | **0.9719** | **0.9411** | **0.9917** | **0.9097** | **0.0103** | **0.0336** |
| Late Fusion Baseline | 0.9497 | 0.9553 | 0.8981 | 0.9774 | 0.9259 | 0.9907 | 0.8888 | 0.1806 | 0.0521 |
| Early Fusion Baseline | 0.9577 | 0.9814 | 0.8963 | 0.9908 | 0.9369 | 0.9937 | 0.9073 | 0.1667 | 0.0418 |
| Text-Only Baseline | 0.9437 | 0.9238 | 0.9144 | 0.9594 | 0.9191 | 0.9860 | 0.8759 | 0.0050 | 0.0445 |
| Image-Only Baseline | 0.7574 | 0.8629 | 0.3648 | 0.9688 | 0.5128 | 0.8089 | 0.4481 | 0.0712 | 0.1782 |
| Audio-Only Baseline | 0.7117 | 0.8468 | 0.2151 | 0.9790 | 0.3431 | 0.7184 | 0.3254 | 0.0825 | 0.2014 |

### Statistical Significance
*   **Proposed CC-AGLF vs. Text-Only**: McNemar $\chi^2 = 186.4$ ($p < 0.0001$)
*   **Proposed CC-AGLF vs. Late Fusion**: McNemar $\chi^2 = 9.12$ ($p = 0.0025$)
*   **95% Confidence Intervals (CI)**: Accuracy $[0.9548, 0.9632]$ | F1-Score $[0.9358, 0.9464]$ (via 1,000 bootstrap resamples)

---

## 🚀 Getting Started

### 1. Installation

Install the necessary dependencies:
```bash
pip install numpy scikit-learn python-docx
```

### 2. Hydrate the Media Assets

Download the raw media assets from public servers using the hydrator utility:
```bash
python hydrate_dataset.py
```

### 3. Run Experimental Baselines

To train the unimodal encoders and run the stacked CC-AGLF pipeline:
```bash
python run_ml_experiment.py
```

---

## ⚖️ License & Citation

The code and manifest are licensed under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International license (CC BY-NC-SA 4.0)**. 

If you use this benchmark in your research, please cite:
```bibtex
@article{fatima2026fusion,
  title={A Fusion-Based Multimodal Framework for Harmful Content Detection in Code-Mixed Roman Urdu Social Media},
  year={2026}
  
}
```
