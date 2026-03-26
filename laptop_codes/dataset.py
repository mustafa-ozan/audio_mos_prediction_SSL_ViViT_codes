import torchaudio
import torch
import soundfile as sf  # Import this directly
import os
import numpy as np
from torch.utils.data import Dataset
from transformers import Wav2Vec2FeatureExtractor
import pandas as pd
from torch.nn.utils.rnn import pad_sequence

import torch
import soundfile as sf  # Import this directly
import os
import numpy as np
from torch.utils.data import Dataset
from transformers import Wav2Vec2FeatureExtractor


class MOSDataset(Dataset):
    def __init__(self, df, wav_base_dir):
        self.df = df
        self.wav_base_dir = wav_base_dir
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-large")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        wav_name = str(self.df.iloc[idx, 0]).strip()
        wav_path = os.path.join(self.wav_base_dir, wav_name)
        label = float(self.df.iloc[idx, 1])

        try:
            # Direct loading with soundfile (bypasses Torchaudio/TorchCodec errors)
            audio, sr = sf.read(wav_path)

            # Soundfile returns (samples, channels). Convert to torch (channels, samples)
            audio = torch.from_numpy(audio).float()
            if len(audio.shape) == 1:
                audio = audio.unsqueeze(0)  # Add channel dim
            else:
                audio = audio.transpose(0, 1)  # (C, L)

            # Standard 16kHz resample for WavLM
            if sr != 16000:
                import torchaudio.transforms as T
                resampler = T.Resample(sr, 16000)
                audio = resampler(audio)

            # Mono conversion
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)

            input_values = self.processor(audio.squeeze(), sampling_rate=16000, return_tensors="pt").input_values
            return input_values.squeeze(), torch.tensor([label], dtype=torch.float32)

        except Exception as e:
            # This ensures your 130k file loop doesn't stop if one file is bad
            return torch.zeros(16000), torch.tensor([label], dtype=torch.float32)

def collate_fn(batch):
    inputs = pad_sequence([item[0] for item in batch], batch_first=True)
    labels = torch.stack([item[1] for item in batch])
    return inputs, labels