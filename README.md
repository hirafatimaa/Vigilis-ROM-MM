# Vigilis-ROM-MM: Multimodal Harmful Content Detection Benchmark for Code-Mixed Roman Urdu

This repository contains the dataset manifest, error analysis, baseline code, and hydration scripts for the paper:
**"A Fusion-Based Multimodal Framework for Harmful Content Detection in Code-Mixed Roman Urdu Social Media"**

---

## 📂 Repository Contents

*   `dataset_manifest_clean.csv`: The index file for the **50,000-sample Vigilis-ROM-MM benchmark corpus**. To comply with platform Terms of Service and GDPR rules, it contains post IDs, platform sources, verified annotations, and split partitions, but no raw media files. Every text entry is 100% unique.
*   `error_analysis_clean.csv`: Detailed log containing the classification errors mapped during baseline evaluations, categorized by failure types.
*   `hydrate_dataset.py`: Python utility script to download and hydrate public media files (text, audio, and visual tracks) locally on your machine.
*   `run_ml_experiment.py`: Main ML pipeline script to train and validate baseline models, ablated configurations, and the proposed CC-AGLF layer.

---

## 📈 Baseline Experimental Results (N_test = 10,000)

Evaluated on the isolated test partition of the 50,000 dataset:

| Model Configuration | Accuracy | Precision | Recall | Specificity | F1-Score | ROC-AUC | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Proposed CC-AGLF** | **0.9577** | **0.9434** | **0.9359** | **0.9695** | **0.9396** | **0.9916** | **0.9071** |
| Late Fusion Baseline (Static) | 0.9504 | 0.9582 | 0.8983 | 0.9787 | 0.9273 | 0.9910 | 0.8908 |
| Early Fusion Baseline | 0.9578 | 0.9814 | 0.8971 | 0.9908 | 0.9374 | 0.9941 | 0.9078 |
| Text-Only Baseline | 0.9447 | 0.9266 | 0.9154 | 0.9606 | 0.9209 | 0.9868 | 0.8785 |
| Image-Only Baseline | 0.7558 | 0.8520 | 0.3699 | 0.9651 | 0.5159 | 0.8050 | 0.4448 |
| Audio-Only Baseline | 0.7115 | 0.8466 | 0.2197 | 0.9784 | 0.3489 | 0.7181 | 0.3284 |

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
*(Note: Edit `hydrate_dataset.py` to point to `dataset_manifest_clean.csv`)*

### 3. Run Experiments
Execute the pipeline runner script to train the unimodal encoders and run stacked CC-AGLF evaluations:
```bash
python run_ml_experiment.py
```

---

## ⚖️ License
The Vigilis-ROM-MM dataset and code are distributed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International license (**CC BY-NC-SA 4.0**) strictly for non-commercial academic research.
