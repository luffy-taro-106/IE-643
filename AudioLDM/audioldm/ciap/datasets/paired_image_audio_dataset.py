# ...existing code...
import torch
from pathlib import Path
from PIL import Image
import soundfile as sf
import numpy as np
import torchvision.transforms as T

class PairedImageAudioDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root,                     # Path or str to dataset folder
        image_ext: str = ".png",
        audio_ext: str = ".wav",
        image_size=(224, 224),
        audio_length: int = 16000,
        transform=None,
    ):
        self.root = Path(root)
        self.image_ext = image_ext
        self.audio_ext = audio_ext
        self.image_size = image_size
        self.audio_length = audio_length
        self.transform = transform or T.Compose([T.Resize(image_size), T.ToTensor()])

        # collect files and pair by stem
        images = sorted(self.root.glob(f"*{self.image_ext}"))
        audios = sorted(self.root.glob(f"*{self.audio_ext}"))
        img_map = {p.stem: p for p in images}
        aud_map = {p.stem: p for p in audios}
        common = sorted(set(img_map.keys()) & set(aud_map.keys()))
        self.pairs = [(img_map[k], aud_map[k]) for k in common]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, aud_path = self.pairs[idx]
        # load image
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)  # tensor [C,H,W], float

        # load audio
        wav, sr = sf.read(str(aud_path))
        if wav.ndim > 1:
            wav = np.mean(wav, axis=1)  # to mono
        # normalize to [-1,1] if needed
        if wav.dtype != np.float32:
            wav = wav.astype(np.float32)
            maxv = np.abs(wav).max() if wav.size else 1.0
            if maxv > 0:
                wav = wav / maxv
        # pad / truncate to audio_length
        if len(wav) < self.audio_length:
            pad = self.audio_length - len(wav)
            wav = np.pad(wav, (0, pad))
        else:
            wav = wav[: self.audio_length]
        audio_tensor = torch.from_numpy(wav).unsqueeze(0)  # [1, T]

        return img, audio_tensor
# ...existing code...