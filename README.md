<div align="center">

# 🎙️ Audio-MOS-Prediction-SSL-ViViT

### A Research-Oriented Toolkit for Subjective Speech Quality Assessment (SSQA)

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-orange)](https://huggingface.co/mustafa-ozan-duman/wavlm-transformer-mos-english)

**Benchmarking the Generalization Abilities of WavLM and ViViT for MOS Prediction**

</div>

📄 [arXiv Paper (2026)](#)

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
