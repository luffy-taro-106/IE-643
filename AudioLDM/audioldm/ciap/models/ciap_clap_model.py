import torch
import torch.nn as nn
from audioldm.clap.open_clip import create_model
from audioldm.clap.training.data import get_audio_features
from audioldm.ciap.models.image_encoder import ImageEncoder
import torchaudio

class CIAP_CLAP_Model(nn.Module):
    """
    CLAP-like wrapper:
      - builds CLAP audio model via create_model (HTSAT/PANN)
      - uses your ImageEncoder as the 'text' branch (image->embedding)
      - exposes get_audio_embedding(audio_dict_list) and get_image_embedding(images_tensor)
    """
    def __init__(self, amodel="HTSAT-tiny", tmodel="roberta", pretrained_path="", image_proj_dim=512, device="cuda"):
        super().__init__()
        self.device = torch.device(device)
        # create CLAP audio model (reuse CLAP factory)
        # create_model returns (model, model_cfg)
        self.model, self.model_cfg = create_model(amodel, tmodel, pretrained_path, precision="fp32", device=self.device, enable_fusion=False)
        # freeze CLAP internals by default (optional finetune)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()
        # image encoder (ResNet18 -> proj dim)
        self.image_encoder = ImageEncoder({"output_dim": image_proj_dim, "pretrained": True})
        self.image_encoder.to(self.device)
        # image projection (if needed to map to same dim as audio)
        self.image_proj = nn.Identity()  # our ImageEncoder already outputs desired dim
        # ensure final dims
        self.embed_dim = image_proj_dim

    def get_audio_embedding(self, audio_dict_list):
        """
        audio_dict_list: list of dicts prepared by get_audio_features(...) as CLAP training expects.
        Returns tensor [B, D]
        """
        with torch.no_grad():
            emb = self.model.get_audio_embedding(audio_dict_list)  # many CLAP models return [B, D]
            # ensure tensor on device
            if not isinstance(emb, torch.Tensor):
                emb = torch.tensor(emb, device=self.device)
            emb = emb.to(self.device).float()
        return emb

    def preprocess_audio_waveform(self, waveform, expected_length=None):
        """
        Helper: waveform [B, T] or [B, 1, T] float32; resample to 48000 if CLAP audio expects 48k
        Returns audio_dict_list via get_audio_features for create_model.get_audio_embedding
        """
        bs = waveform.size(0)
        sr = 16000  # your dataset sr
        # many CLAP audio models expect 48000; resample if needed to 48000
        target_sr = self.model_cfg.get("audio_cfg", {}).get("target_sample_rate", 48000) if hasattr(self, "model_cfg") else 48000
        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=target_sr)
        audio_dicts = []
        for w in waveform:
            audio_dict = {}
            audio_dict = get_audio_features(audio_dict, w, target_sr * 10, data_truncating="fusion", data_filling="repeatpad", audio_cfg=self.model_cfg.get("audio_cfg", {}))
            audio_dicts.append(audio_dict)
        return audio_dicts

    def get_image_embedding(self, images):
        """
        images: [B, C, H, W] torch tensor
        Returns [B, D] float tensor on device
        """
        images = images.to(self.device)
        # allow gradients so the image encoder is trainable
        emb = self.image_encoder(images)   # [B, D], will have grad when encoder params require_grad=True
        emb = emb.to(self.device).float()
        return emb

    def to(self, device):
        self.device = torch.device(device)
        self.model.to(self.device)
        self.image_encoder.to(self.device)
        return self