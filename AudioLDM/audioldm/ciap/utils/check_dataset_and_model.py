import torch
from audioldm.ciap.datasets.paired_image_audio_dataset import PairedImageAudioDataset
from audioldm.ciap.models.image_encoder import ImageEncoder
from audioldm.ciap.models.audio_encoder import AudioEncoder
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # dataset (paths come from your ciap_config.yaml defaults)
    ds = PairedImageAudioDataset(root=Path.cwd()/"data"/"train",
                                 image_ext=".png", audio_ext=".wav")
    print("dataset size:", len(ds))
    img, aud = ds[0][:2]  # adapt if dataset returns more
    print("sample image shape:", getattr(img, "shape", None))
    print("sample audio shape:", getattr(aud, "shape", None))

    # instantiate encoders
    img_enc = ImageEncoder()
    aud_enc = AudioEncoder()
    img_enc.to(device).eval()
    aud_enc.to(device).eval()

    # make batch
    img = img.unsqueeze(0).to(device)
    aud = aud.unsqueeze(0).to(device)

    with torch.no_grad():
        img_feat = img_enc.encode(img)
        aud_feat = aud_enc.encode(aud)
    print("image embedding:", img_feat.shape if hasattr(img_feat, "shape") else type(img_feat))
    print("audio embedding:", aud_feat.shape if hasattr(aud_feat, "shape") else type(aud_feat))

if __name__ == "__main__":
    main()