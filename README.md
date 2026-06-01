# Vigilis-ROM-MM: Multimodal Harmful Content Detection Benchmark for Code-Mixed Roman Urdu

This repository contains the dataset manifest, error analysis, baseline code, and hydration scripts for the paper:
**"A Fusion-Based Multimodal Framework for Harmful Content Detection in Code-Mixed Roman Urdu Social Media"**

---

## 📂 Repository Contents

*   `dataset_manifest.csv`: The index file for the **50,000-sample Vigilis-ROM-MM benchmark corpus**. To comply with platform Terms of Service and GDPR rules, it contains post IDs, platform sources, verified annotations, and split partitions, but no raw media files. Every text entry is 100% unique.
*   `error_analysis.csv`: Detailed log containing the classification errors mapped during baseline evaluations, categorized by failure types.
*   `hydrate_dataset.py`: Python utility script to download and hydrate public media files (text, audio, and visual tracks) locally on your machine.
*   `run_ml_experiment.py`: Main ML pipeline script to train and validate baseline models, ablated configurations, and the proposed CC-AGLF layer.

---

## 📈 Baseline Experimental Results (N_test = 10,000)

Evaluated on the isolated test partition of the 50,000 dataset:

| Model Configuration | Accuracy | Precision | Recall | Specificity | F1-Score | ROC-AUC | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Proposed CC-AGLF** | **0.9590** | **0.9472** | **0.9351** | **0.9719** | **0.9411** | **0.9917** | **0.9097** |
| Late Fusion Baseline (Static) | 0.9497 | 0.9553 | 0.8981 | 0.9774 | 0.9259 | 0.9907 | 0.8888 |
| Early Fusion Baseline | 0.9577 | 0.9814 | 0.8963 | 0.9908 | 0.9369 | 0.9937 | 0.9073 |
| Text-Only Baseline | 0.9437 | 0.9238 | 0.9144 | 0.9594 | 0.9191 | 0.9860 | 0.8759 |
| Image-Only Baseline | 0.7574 | 0.8629 | 0.3648 | 0.9688 | 0.5128 | 0.8089 | 0.4481 |
| Audio-Only Baseline | 0.7117 | 0.8468 | 0.2151 | 0.9790 | 0.3431 | 0.7184 | 0.3254 |

---

## 🚀 Getting Started

### 1. Installation
Install the necessary python dependencies:
```bash
pip install numpy scikit-learn
```

### 2. Hydrate the Media Assets
Execute the downloader script to download raw media files locally based on the post IDs in the manifest:
```bash
python hydrate_dataset.py
```
*(Note: Edit `hydrate_dataset.py` to point to `dataset_manifest.csv`)*

### 3. Run Experiments
Execute the pipeline runner script to train the unimodal encoders and run stacked CC-AGLF evaluations:
```bash
python run_ml_experiment.py
```

---

## ⚖️ License
The Vigilis-ROM-MM dataset and code are distributed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International license (**CC BY-NC-SA 4.0**) strictly for non-commercial academic research.
