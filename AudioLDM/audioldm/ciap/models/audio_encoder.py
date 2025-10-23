# ...existing code...
import torch
import torch.nn as nn
from typing import Union, Dict

class AudioEncoder(torch.nn.Module):
    def __init__(self, config: Union[int, Dict]=16000, hidden_dim: int = 1024, output_dim: int = 512):
        """
        Accept either:
         - input_dim (int) as first arg
         - config dict with keys: input_size (list or int), hidden_dim, output_size/output_dim
        """
        super(AudioEncoder, self).__init__()
        if isinstance(config, dict):
            # config may specify input_size as [1, 16000] or input_dim directly
            input_spec = config.get("input_size", config.get("input_dim", config))
            if isinstance(input_spec, (list, tuple)):
                input_dim = int(input_spec[-1])
            else:
                input_dim = int(input_spec)
            hidden_dim = int(config.get("hidden_dim", hidden_dim))
            output_dim = int(config.get("output_size", config.get("output_dim", output_dim)))
        else:
            input_dim = int(config)

        self.input_dim = input_dim
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        # accept [B, T] or [B, 1, T] or [B, C, T]
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def encode(self, audio_samples):
        return self.forward(audio_samples)

    def extract_features(self, audio_samples):
        if audio_samples.dim() > 2:
            audio_samples = audio_samples.view(audio_samples.size(0), -1)
        return self.fc1(audio_samples)
# ...existing code...