<div align="center">

# 🎙️ Audio-MOS-Prediction-SSL-ViViT

### A Research-Oriented Toolkit for Subjective Speech Quality Assessment (SSQA)

[![Python 3.12](https://shields.io)](https://python.org)
[![License](https://shields.io)](https://opensource.org)
[![arXiv](https://shields.io)](https://doi.org)
[![HuggingFace](https://shields.io)](https://huggingface.co)

**Benchmarking the Generalization Abilities of WavLM and ViViT for MOS Prediction**

</div>

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

## 📝 Citation

If you find this benchmark or the provided scripts helpful for your research, please cite our study:

```bibtex
@article{duman2026audio,
  title={Audio MOS Prediction using SSL-based Transformers and ViViT adaptation},
  author={Duman, Mustafa Ozan},
  journal={arXiv preprint arXiv:2607.10146},
  year={2026},
  eprint={2607.10146},
  archivePrefix={arXiv},
  primaryClass={eess.AS},
  url={https://doi.org}
}
```

<div align="center">

**Mustafa Ozan Duman**  
Research Assistant, Bursa Uludag University  
Computer Engineering Department, Turkey  

</div>
