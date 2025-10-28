import os
import yaml
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from audioldm.ciap.datasets.paired_image_audio_dataset import PairedImageAudioDataset
from audioldm.ciap.models.image_encoder import ImageEncoder
from audioldm.ciap.models.audio_encoder import AudioEncoder
from audioldm.ciap.losses.contrastive_loss import ContrastiveLoss
from tqdm import tqdm

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def train_contrastive_model(config_path, cli_args=None):
    config = load_config(config_path)

    # dataset path (support both config layouts)
    dataset_cfg = config.get("dataset", {})
    if isinstance(dataset_cfg.get("train"), dict) and "path" in dataset_cfg.get("train"):
        dataset_path = dataset_cfg["train"]["path"]
        image_ext = dataset_cfg["train"].get("image_extension", ".jpg")
        audio_ext = dataset_cfg["train"].get("audio_extension", ".wav")
    elif "path" in dataset_cfg:
        dataset_path = dataset_cfg["path"]
        image_ext = dataset_cfg.get("image_extension", ".jpg")
        audio_ext = dataset_cfg.get("audio_extension", ".wav")
    else:
        raise KeyError("Could not find dataset path in config. Expected dataset.train.path or dataset.path")

    # CLI overrides
    device_str = None
    if cli_args and getattr(cli_args, "device", None):
        device_str = cli_args.device
    else:
        device_str = config.get("device", {}).get("type", "cpu") if isinstance(config.get("device", {}), dict) else config.get("device", "cpu")
    device = torch.device(device_str)

    batch_size = cli_args.batch_size if (cli_args and cli_args.batch_size) else config["training"].get("batch_size", 32)
    num_epochs = config["training"].get("num_epochs", 10)
    lr = config["training"].get("learning_rate", 1e-3)

    output_dir = cli_args.output_dir if (cli_args and cli_args.output_dir) else config.get("logging", {}).get("log_dir", "ckpt/ciap")
    os.makedirs(output_dir, exist_ok=True)

    # dataset + dataloader
    dataset = PairedImageAudioDataset(dataset_path, image_ext=image_ext, audio_ext=audio_ext)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # models
    image_encoder = ImageEncoder(config["model"]["image_encoder"])
    audio_encoder = AudioEncoder(config["model"]["audio_encoder"])
    image_encoder.to(device)
    audio_encoder.to(device)
    image_encoder.train()
    audio_encoder.train()

    # optimizer + loss
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(list(image_encoder.parameters()) + list(audio_encoder.parameters()), lr=lr)

    # training loop with tqdm
    for epoch in range(1, num_epochs + 1):
        running_loss = 0.0
        loop = tqdm(enumerate(dataloader, start=1), total=len(dataloader),
                    desc=f"Epoch {epoch}/{num_epochs}", leave=True)
        for batch_idx, (images, audios) in loop:
            images = images.to(device)
            audios = audios.to(device)

            optimizer.zero_grad()

            # use encode() if available; fallback to forward()
            img_emb = image_encoder.encode(images) if hasattr(image_encoder, "encode") else image_encoder(images)
            aud_emb = audio_encoder.encode(audios) if hasattr(audio_encoder, "encode") else audio_encoder(audios)

            # labels: i-th image matches i-th audio
            labels = torch.arange(img_emb.size(0), device=device)

            loss = criterion(img_emb, aud_emb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            avg_loss = running_loss / batch_idx
            loop.set_postfix({"avg_loss": f"{avg_loss:.4f}", "batch_loss": f"{loss.item():.4f}"})

        print(f"Epoch {epoch} finished. EpochLoss: {running_loss/len(dataloader):.4f}")

    # save final models
    img_path = os.path.join(output_dir, "ciap_image_encoder.pt")
    aud_path = os.path.join(output_dir, "ciap_audio_encoder.pt")
    torch.save(image_encoder.state_dict(), img_path)
    torch.save(audio_encoder.state_dict(), aud_path)
    print(f"Saved image encoder -> {img_path}")
    print(f"Saved audio encoder  -> {aud_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CIAP contrastive model")
    parser.add_argument("--config", type=str, default="audioldm/ciap/configs/ciap_config.yaml")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")

    train_contrastive_model(args.config, cli_args=args)
