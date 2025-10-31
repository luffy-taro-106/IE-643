import torch
import torch.nn as nn

class CIAPCondStage(nn.Module):
    """
    CLAP-like wrapper for CIAP encoders.

    Methods:
      - encode(x) -> [B, D] (image) normalized
      - audio_to_embedding(audio) -> [B, D] normalized
      - get_unconditional_condition(batch_size) -> zeros [B, D]
      - cos_similarity(a, b) -> numpy [N, M] cosine similarities
      - to(device), eval(), train()
    """
    def __init__(self, image_encoder, audio_encoder, embed_dim=512, device="cpu"):
        super().__init__()
        self.image_encoder = image_encoder
        self.audio_encoder = audio_encoder
        self.embed_dim = embed_dim
        self.device = torch.device(device)

        # move models and set eval
        self.image_encoder.to(self.device)
        self.audio_encoder.to(self.device)
        self.image_encoder.eval()
        self.audio_encoder.eval()

    def to(self, device):
        self.device = torch.device(device)
        self.image_encoder.to(self.device)
        self.audio_encoder.to(self.device)
        return self

    def eval(self):
        self.image_encoder.eval()
        self.audio_encoder.eval()
        return self

    def train(self, mode=True):
        self.image_encoder.train(mode)
        self.audio_encoder.train(mode)
        return self

    def _call_encoder(self, encoder, x):
        # safe encode wrapper: accepts raw tensors of shape [B, C, H, W] for images
        # or [B, T] / [B, 1, T] for audio. Returns L2-normalized [B, D] float tensor on device.
        with torch.no_grad():
            x = x.to(self.device)
            # some audio encoders expect flattened waveform; keep as-is and rely on audio_encoder.encode
            if hasattr(encoder, "encode"):
                out = encoder.encode(x)
            else:
                out = encoder(x)
            out = out.to(self.device).float()
            # If output has extra dims (e.g., [B,1,D]), squeeze
            out = out.view(out.size(0), -1) if out.dim() > 2 else out
            # L2-normalize
            out = out / (out.norm(dim=-1, keepdim=True) + 1e-8)
            return out

    def encode(self, x):
        """Encode images -> [B, D] normalized"""
        return self._call_encoder(self.image_encoder, x)

    def get_unconditional_condition(self, batch_size):
        return torch.zeros(batch_size, self.embed_dim, device=self.device, dtype=torch.float32)

    def audio_to_embedding(self, audio_tensor):
        """Encode raw audio waveform -> [B, D] normalized"""
        return self._call_encoder(self.audio_encoder, audio_tensor)

    def cos_similarity(self, waves, images_or_texts):
        """Return numpy cosine similarity matrix [N, M]"""
        if waves.dim() > 2 or waves.shape[-1] > self.embed_dim:
            waves = self.audio_to_embedding(waves)
        if images_or_texts.dim() > 2 or images_or_texts.shape[-1] > self.embed_dim:
            images_or_texts = self.encode(images_or_texts)
        waves = waves / (waves.norm(dim=-1, keepdim=True) + 1e-8)
        images_or_texts = images_or_texts / (images_or_texts.norm(dim=-1, keepdim=True) + 1e-8)
        sim = torch.matmul(waves, images_or_texts.t()).cpu().numpy()
        return sim