Audio MOS Prediction: SSL and ViViT Experiments

This repository contains the source code, training pipelines, and feature extraction utilities developed for predicting Mean Opinion Score (MOS). The project explores advanced deep learning architectures, specifically focusing on Self-Supervised Learning (SSL) models like WavLM and Video Vision Transformers (ViViT) adapted for audio quality assessment.

Our goal is to provide a comprehensive guide and codebase to assist researchers in the field of speech processing and audio quality evaluation.

🚀 Live Predictor
For those looking to use the pre-trained model directly without setting up the training environment, we have released a "plug-and-play" version here:
👉 Hugging Face: WavLM-Transformer-MOS-English

📂 Repository Structure
The project is divided into two main sections based on the workflow used during research:

1. laptop_codes/ (Local Environment)
Contains production-level scripts for final model training and high-performance inference.

model.py & dataset.py: The core architecture and data handling logic.

train_SSL_frozen.py: Optimized script for training with frozen SSL backbones.

requirements.txt: Complete environment specification for local reproduction.

2. colab_codes/ (Cloud/Experimental)
Contains Jupyter notebooks used for initial feature engineering and cloud-based experimentation.

MFCC_EXTRACTOR.ipynb: Comprehensive guide for extracting MFCC features for baseline studies.

SSL_Transformer.ipynb: Experiments involving SSL-based feature aggregation.

ViViT_Transformer.ipynb: Novel adaptation of Vision Transformers for audio spectrogram analysis.

🛠️ Getting Started
Local Setup
To run the training scripts on your local machine:

Clone the repository:

Bash
git clone https://github.com/mustafa-ozan/audio_mos_prediction_SSL_ViViT_codes.git
Set up your environment using the provided requirements:

Bash
pip install -r laptop_codes/requirements.txt
Hardware Recommendations
Python: 3.12

GPU: NVIDIA RTX series (e.g., 5070 Ti) is recommended for training, though the inference scripts support CPU execution.

📚 Reference & Citation
If you find this codebase or the trained model helpful for your research, please cite our work:

Mustafa Ozan Duman, Bursa Uludag University, Computer Engineering Department.
[Insert Article Title Here]
[Insert Journal/Conference Name, Year]
Link: [Link to Article/DOI]

License: Apache 2.0

Affiliation: Bursa Uludag University, Computer Engineering Department.
