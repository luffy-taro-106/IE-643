import torch
import torch.nn as nn
import numpy as np

class CIAPCondStage(nn.Module):
    """
    Minimal wrapper providing CLAP-like API:
      - encode(input) -> embedding tensor [B, D]
      - get_unconditional_condition(batch_size) -> zeros [B, D]
      - cos_similarity(waves, images_or_texts) -> numpy array similarities
      - to(device), eval(), train()
    Assumes you pass pre-instantiated image_encoder and audio_encoder,
    each exposing encode(tensor) -> tensor [B, D].
    """
    def __init__(self, image_encoder, audio_encoder, embed_dim=512, device="cpu"):
        super().__init__()
        self.image_encoder = image_encoder
        self.audio_encoder = audio_encoder
        self.embed_dim = embed_dim
        self.device = torch.device(device)
        self.to(self.device)
        self.eval()

    def to(self, device):
        self.device = torch.device(device)
        self.image_encoder.to(self.device)
        self.audio_encoder.to(self.device)
        return super().to(self.device)

    def eval(self):
        self.image_encoder.eval()
        self.audio_encoder.eval()
        return super().eval()

    def train(self, mode=True):
        self.image_encoder.train(mode)
        self.audio_encoder.train(mode)
        return super().train(mode)

    def encode(self, x):
        """
        Accepts:
          - tensor image: [B, 3, H, W]
          - list of images: list of tensors
        Returns: torch.Tensor [B, D] on same device
        """
        if isinstance(x, list):
            x = torch.stack([xi.to(self.device) for xi in x], dim=0)
        else:
            x = x.to(self.device)
        with torch.no_grad():
            emb = self.image_encoder.encode(x)
        return emb.to(self.device)

    def get_unconditional_condition(self, batch_size):
        # simple zero unconditional embedding (same shape as encode output)
        return torch.zeros(batch_size, self.embed_dim, device=self.device)

    def audio_to_embedding(self, audio_tensor):
        # accepts numpy or torch; returns torch [B, D]
        if isinstance(audio_tensor, np.ndarray):
            audio_tensor = torch.from_numpy(audio_tensor).float()
        audio_tensor = audio_tensor.to(self.device)
        with torch.no_grad():
            emb = self.audio_encoder.encode(audio_tensor)
        return emb

    def cos_similarity(self, waves, images_or_texts):
        """
        Compute cosine similarity between waves and images (or image embeddings).
        - waves: numpy array [Nw, T] or torch tensor [Nw, 1, T]
        - images_or_texts: list or tensor of images or precomputed embeddings
        Returns: numpy array shape (Nw, Ni)
        """
        # audio -> emb
        if isinstance(waves, np.ndarray):
            aud = torch.from_numpy(waves).float()
        else:
            aud = waves
        aud = aud.to(self.device)
        with torch.no_grad():
            a_emb = self.audio_encoder.encode(aud)
        # images_or_texts -> emb
        if isinstance(images_or_texts, np.ndarray) or isinstance(images_or_texts, torch.Tensor):
            img_input = images_or_texts
            if isinstance(img_input, np.ndarray):
                img_input = torch.from_numpy(img_input).float()
            img_input = img_input.to(self.device)
            with torch.no_grad():
                i_emb = self.image_encoder.encode(img_input)
        elif isinstance(images_or_texts, list):
            # assume list of image tensors
            img_input = torch.stack([it.to(self.device) for it in images_or_texts])
            with torch.no_grad():
                i_emb = self.image_encoder.encode(img_input)
        else:
            # can't handle other types, return zeros
            return np.zeros((a_emb.shape[0], 0))

        # normalize and compute cosine
        a_norm = a_emb / (a_emb.norm(dim=1, keepdim=True) + 1e-8)
        i_norm = i_emb / (i_emb.norm(dim=1, keepdim=True) + 1e-8)
        sim = torch.matmul(a_norm, i_norm.t()).cpu().numpy()
        return sim