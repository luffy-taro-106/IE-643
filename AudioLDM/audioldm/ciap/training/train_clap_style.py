import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from audioldm.ciap.models.ciap_clap_model import CIAP_CLAP_Model
from audioldm.ciap.datasets.paired_image_audio_dataset import PairedImageAudioDataset
from audioldm.ciap.losses.contrastive_loss import contrastive_loss  # assume this returns scalar
from audioldm.clap.training.data import get_audio_features
import argparse
import math
import torchaudio

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CIAP_CLAP_Model(amodel=args.amodel, tmodel="roberta", pretrained_path=args.pretrained_audio_ckpt, image_proj_dim=args.embed_dim, device=device)
    model.to(device)
    model.train()  # set image proj to train if present

    # prepare dataset
    dataset = PairedImageAudioDataset(args.dataset_path, image_ext=args.image_ext, audio_ext=args.audio_ext)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # trainable params: image encoder projection head (and optionally fine-tune audio projection)
    params = list(model.image_encoder.parameters())
    # optionally fine-tune audio projection (if you add a small head)
    optimizer = optim.Adam(params, lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    for epoch in range(args.epochs):
        total_loss = 0.0
        for batch_idx, (image, audio, path) in enumerate(loader):
            image = image.to(device)          # [B, C, H, W], expected normalized already by dataset
            audio = audio.to(device)          # [B, 1, T] or [B, T]

            # prepare CLAP audio inputs
            audio_dicts = model.preprocess_audio_waveform(audio)
            # get embeddings
            aud_emb = model.get_audio_embedding(audio_dicts)      # [B, D]
            img_emb = model.get_image_embedding(image)            # [B, D]

            # normalize
            aud_emb = aud_emb / (aud_emb.norm(dim=-1, keepdim=True) + 1e-8)
            img_emb = img_emb / (img_emb.norm(dim=-1, keepdim=True) + 1e-8)

            # contrastive loss (InfoNCE-like)
            loss = contrastive_loss(img_emb, aud_emb, temperature=0.07)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if batch_idx % args.log_interval == 0:
                print(f"Epoch {epoch+1} [{batch_idx}/{len(loader)}] loss: {loss.item():.4f}")

        scheduler.step()
        print(f"Epoch {epoch+1} avg loss: {total_loss / len(loader):.4f}")

        # save checkpoint
        ckpt_dir = args.save_dir
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(model.image_encoder.state_dict(), os.path.join(ckpt_dir, f"ciap_image_encoder_epoch{epoch+1}.pt"))

    print("Training finished.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='./data/train', help='paired dataset root')
    parser.add_argument('--image_ext', type=str, default='.png')
    parser.add_argument('--audio_ext', type=str, default='.wav')
    parser.add_argument('--amodel', type=str, default='HTSAT-tiny')
    parser.add_argument('--pretrained_audio_ckpt', type=str, default='')
    parser.add_argument('--embed_dim', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--save_dir', type=str, default='./ckpt')
    args = parser.parse_args()
    train(args)