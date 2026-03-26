<div align="center">

# 🎙️ Audio-MOS-Prediction-SSL-ViViT

### A Research-Oriented Toolkit for Subjective Speech Quality Assessment (SSQA)

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-orange)](https://huggingface.co/mustafa-ozan-duman/wavlm-transformer-mos-english)

**Benchmarking the Generalization Abilities of WavLM and ViViT for MOS Prediction**

</div>

---

* **MOS-Prediction** is a benchmark designed to evaluate how well self-supervised learning (SSL) models perceive audio quality as humans do.
* **SSL-ViViT Framework** stands for the **S**peech **S**elf-supervised **L**earning and **Vi**sual **Vi**deo **T**ransformer toolkit, designed for advanced MOS estimation research.

📚 [Full Documentation (Coming Soon)](#) | 📄 [arXiv Paper (2026)](#) | 🤗 [Hugging Face Space Demo](https://huggingface.co/mustafa-ozan-duman/wavlm-transformer-mos-english)

---

## 🔍 Overview

This repository provides the complete experimental codebase for feature extraction and model training. The project compares traditional acoustic features (MFCC) with state-of-the-art SSL features (WavLM) to find the most effective representation for human-like speech quality perception.

* **Key Models:** WavLM-Transformer, ViViT-Audio-Regressor.
* **Objective:** High-accuracy MOS prediction on a 1.0 - 5.0 scale.
* **Research Phase:** Developed during PhD candidacy at Bursa Uludag University.

---

## 📂 Repository Structure

The project is organized by execution environment to ensure reproducibility across local and cloud platforms:

| Directory | Description |
| :--- | :--- |
| [**`laptop_codes/`**](./laptop_codes/) | **Production Core:** Python scripts for local training, dataset handling, and batch inference. |
| [**`colab_codes/`**](./colab_codes/) | **Experimental Lab:** Jupyter Notebooks for MFCC extraction, ViViT adaptations, and SSL trials. |

---

## 🛠️ Usage Guide

### 1. Environment Setup
The project is built on **Python 3.12**. To set up your local environment:
```bash
# Clone the repository
git clone [https://github.com/mustafa-ozan/audio_mos_prediction_SSL_ViViT_codes.git](https://github.com/mustafa-ozan/audio_mos_prediction_SSL_ViViT_codes.git)
cd audio_mos_prediction_SSL_ViViT_codes/laptop_codes

# Install dependencies
pip install -r requirements.txt
2. Batch MOS Label Generation
To generate quality labels for a directory of audio files using the pre-trained WavLM-Transformer:

Bash
python predict_folder.py --dir /path/to/your/wav_files --out mos_results.csv
📈 Benchmarking Results
See our Evaluation Logs for detailed performance metrics across various test sets.

March 2026: Achieved SOTA results on English-only MOS sets using WavLM-Base.

February 2026: Released the first iteration of the ViViT-Transformer for audio spectrograms.

📝 Citation
If you find this benchmark or the provided scripts helpful for your research, please cite our study:

Kod snippet'i
@article{duman2026audio,
  title={Audio MOS Prediction using SSL-based Transformers and ViViT adaptation},
  author={Duman, Mustafa Ozan},
  journal={Bursa Uludag University Computer Engineering Research},
  year={2026},
  url={[https://github.com/mustafa-ozan/audio_mos_prediction_SSL_ViViT_codes](https://github.com/mustafa-ozan/audio_mos_prediction_SSL_ViViT_codes)}
}
<div align="center">

Mustafa Ozan Duman Research Assistant, Bursa Uludag University

Computer Engineering Department, Turkey

</div>
