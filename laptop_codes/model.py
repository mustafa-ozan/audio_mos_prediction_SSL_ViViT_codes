import torch
import torch.nn as nn
from transformers import WavLMModel

class WavLMTransformerMOS(nn.Module):
    def __init__(self, freeze_wavlm=True):
        super().__init__()
        self.wavlm = WavLMModel.from_pretrained("microsoft/wavlm-large")

        if freeze_wavlm:
            for param in self.wavlm.parameters():
                param.requires_grad = False
            print("❄️ WavLM Backbone Frozen.")

        # Transformer Head (d_model=1024 for WavLM-Large)
        encoder_layer = nn.TransformerEncoderLayer(d_model=1024, nhead=8, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.regression = nn.Linear(1024, 1)

    def forward(self, input_values):
        outputs = self.wavlm(input_values).last_hidden_state
        transformed = self.transformer_encoder(outputs)
        pooled = torch.mean(transformed, dim=1) # Global Average Pooling
        return self.regression(pooled)