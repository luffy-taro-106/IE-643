# Image-to-Audio Generation using Contrastive Image-Audio Pretraining (CIAP)

A project that extends AudioLDM's text-to-audio generation capability to image-to-audio generation by replacing the CLAP (Contrastive Language-Audio Pretraining) model with a custom CIAP (Contrastive Image-Audio Pretraining) model that learns to map images and audio into a shared latent space.

## 📋 Project Overview

### Objective
Transform a pretrained text-to-audio generation model (AudioLDM) into an image-to-audio generation model by developing a contrastive learning framework that aligns images and audio in a shared embedding space.

### Key Constraints
- ✅ **Baseline Network (AudioLDM) remains frozen** - No training or fine-tuning allowed on the diffusion model
- ✅ **Only CIAP components are trainable** - Image and audio encoders are trained from scratch
- ✅ **Uses AudioLDM as instructed baseline** - Leverages the provided pretrained diffusion model

## 🏗️ Architecture

### System Overview

```
Input Image → Image Encoder → Image Embedding (512-d)
                                      ↓
                              [Contrastive Alignment]
                                      ↓
AudioLDM Diffusion Model ← Audio Embedding (512-d) ← Audio Encoder
                                      ↓
                              Generated Audio
```

### Components

#### 1. **CIAP Model (Contrastive Image-Audio Pretraining)**

The CIAP model consists of two encoders that learn to map images and audio into a shared 512-dimensional embedding space:

- **Image Encoder**: 
  - Backbone: ResNet18 (pretrained on ImageNet)
  - Architecture: ResNet18 features → Linear projection → LayerNorm
  - Input: RGB images (224×224)
  - Output: 512-dimensional normalized embeddings

- **Audio Encoder**:
  - Architecture: MLP with two linear layers
  - Input: Audio waveforms (16kHz, 1 second segments)
  - Output: 512-dimensional normalized embeddings

- **Contrastive Learning**:
  - Uses InfoNCE loss (similar to CLIP/CLAP)
  - Learns to maximize similarity between paired image-audio embeddings
  - Minimizes similarity between unpaired samples
  - Temperature-scaled cosine similarity

#### 2. **Integration with AudioLDM**

- Replaces the CLAP text encoder with CIAP image encoder
- The image embeddings serve as conditioning for the frozen AudioLDM diffusion model
- Zero-shot inference: Generate audio directly from images without fine-tuning the diffusion model

## 📁 Project Structure

```
AudioLDM/
├── audioldm/
│   ├── ciap/                    # CIAP model implementation
│   │   ├── models/
│   │   │   ├── image_encoder.py      # ResNet18-based image encoder
│   │   │   ├── audio_encoder.py      # MLP-based audio encoder
│   │   │   ├── ciap_cond.py          # Conditioning stage wrapper
│   │   │   └── ciap_clap_model.py    # CLAP-style wrapper
│   │   ├── datasets/
│   │   │   └── paired_image_audio_dataset.py  # Dataset loader
│   │   ├── losses/
│   │   │   └── contrastive_loss.py           # Contrastive loss
│   │   ├── training/
│   │   │   ├── train_contrastive.py          # Training script
│   │   │   └── train_clap_style.py          # Alternative training
│   │   └── configs/
│   │       └── ciap_config.yaml              # Configuration file
│   └── ...                      # AudioLDM baseline (frozen)
├── ckpt/                        # Model checkpoints
│   ├── audioldm-s-full.ckpt              # Frozen AudioLDM baseline
│   ├── ciap_image_encoder_epoch110.pt    # Trained image encoder
│   └── ciap_audio_encoder2.pt            # Trained audio encoder
├── data/                        # Training/validation datasets
│   ├── train/                   # Paired image-audio training data
│   └── val/                     # Paired image-audio validation data
├── image_to_audio_ui.py        # Gradio web interface
└── Inference.ipynb             # Inference notebook
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- PyTorch (CPU or GPU)
- Required packages:
  ```bash
  pip install torch torchvision torchaudio
  pip install gradio pillow numpy soundfile yaml
  pip install matplotlib tqdm
  ```

### Running the Web Interface

1. **Navigate to the AudioLDM directory:**
   ```bash
   cd AudioLDM
   ```

2. **Launch the Gradio UI:**
   ```bash
   python image_to_audio_ui.py
   ```

3. **Access the interface:**
   - Open your browser and navigate to `http://localhost:7860`
   - The UI will automatically load the models on first use

4. **Generate audio from images:**
   - Upload an image (JPG, PNG, or other formats)
   - Adjust the duration slider (2.5-10 seconds)
   - Click "Generate Audio 🎵"
   - Wait for generation to complete (progress bar will show status)
   - Play the generated audio directly in the browser

### Features of the UI

- ✅ **Automatic Image Preprocessing**: Handles various image formats and dimensions
- ✅ **Real-time Progress Tracking**: Shows generation progress with status updates
- ✅ **Audio Playback**: Built-in audio player in the browser
- ✅ **Flexible Duration**: Adjustable audio length (2.5-10 seconds)

## 🔬 Technical Details

### Training Process

1. **Dataset Preparation**:
   - Collect paired image-audio data
   - Each sample contains an image and corresponding audio file
   - Images are preprocessed to 224×224 RGB
   - Audio is segmented to 1-second clips at 16kHz

2. **Contrastive Learning**:
   - Train image and audio encoders jointly
   - Use InfoNCE contrastive loss
   - Maximize similarity for positive pairs (matched image-audio)
   - Minimize similarity for negative pairs (random combinations)

3. **Training Configuration**:
   - Batch size: 32
   - Learning rate: 0.001
   - Optimizer: Adam
   - Scheduler: StepLR (decay every 10 epochs)

### Inference Process

1. **Image Encoding**:
   - Preprocess image: Resize to 224×224, convert to tensor
   - Pass through Image Encoder → 512-d embedding

2. **Audio Generation**:
   - Use image embedding as condition for AudioLDM
   - Run diffusion sampling (1000 steps)
   - Decode latent to mel-spectrogram
   - Convert mel-spectrogram to waveform

3. **Post-processing**:
   - Crop to desired duration
   - Save as WAV file

## 📊 Model Files

### Checkpoints

- **AudioLDM Baseline** (`ckpt/audioldm-s-full.ckpt`): 
  - Frozen pretrained diffusion model
  - Used for audio generation only

- **CIAP Encoders**:
  - `ckpt/ciap_image_encoder_epoch110.pt`: Trained image encoder
  - `ckpt/ciap_audio_encoder2.pt`: Trained audio encoder

### Configuration

- Model configuration: `audioldm/ciap/configs/ciap_config.yaml`
- Image encoder: ResNet18 → 512-d projection
- Audio encoder: MLP (16000 → 1024 → 512)

## 🔍 Key Innovation

The main contribution is replacing the text-based conditioning (CLAP) with image-based conditioning (CIAP) while keeping the AudioLDM diffusion model completely frozen. This demonstrates:

1. **Modular Design**: The conditioning mechanism is separable from the generative model
2. **Zero-shot Transfer**: The pretrained diffusion model can work with new conditioning types without retraining
3. **Efficient Training**: Only the encoders need training, reducing computational cost

## 📝 Usage Example

### Python API

```python
from audioldm import build_model, save_wave
from audioldm.ciap.models.image_encoder import ImageEncoder
from audioldm.ciap.models.audio_encoder import AudioEncoder
from audioldm.ciap.models.ciap_cond import CIAPCondStage
from PIL import Image
import torchvision.transforms as T

# Load models
audioldm = build_model(ckpt_path="./ckpt/audioldm-s-full.ckpt")
image_encoder = ImageEncoder(config).load_state_dict(torch.load("ckpt/ciap_image_encoder_epoch110.pt"))
audio_encoder = AudioEncoder(config).load_state_dict(torch.load("ckpt/ciap_audio_encoder2.pt"))
cond = CIAPCondStage(image_encoder, audio_encoder, embed_dim=512, device="cuda")

# Preprocess image
transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])
image = transform(Image.open("your_image.jpg")).unsqueeze(0).to("cuda")

# Encode and generate
with torch.no_grad():
    img_emb = cond.encode(image)
    waveform_latent = audioldm.sample(cond=img_emb, batch_size=1)
    mel = audioldm.decode_first_stage(waveform_latent)
    waveform = audioldm.mel_spectrogram_to_waveform(mel)

# Save audio
save_wave(waveform, savepath="./output", name="generated_audio")
```

## 🎯 Project Compliance

This project adheres to all specified requirements:

- ✅ Uses AudioLDM as the instructed baseline model
- ✅ Baseline network (AudioLDM diffusion model) remains completely frozen
- ✅ Only CIAP components (image and audio encoders) are trained
- ✅ No modification to the AudioLDM architecture or weights
- ✅ Dataset collected and verified with TAs
- ✅ Demonstrates image-to-audio generation capability

## 🛠️ Development

### Training CIAP Models

To train the CIAP encoders:

```bash
cd AudioLDM
python audioldm/ciap/training/train_contrastive.py --config audioldm/ciap/configs/ciap_config.yaml
```

### Evaluation

Use the Jupyter notebook `Inference.ipynb` for detailed inference and evaluation.

## 📚 References

- **AudioLDM**: [AudioLDM: Text-to-Audio Generation with Latent Diffusion Models](https://arxiv.org/abs/2301.12503)
- **CLAP**: Contrastive Language-Audio Pretraining (used as reference architecture)
- **CLIP**: Contrastive Language-Image Pretraining (inspiration for contrastive learning approach)

## 👥 Acknowledgments

- AudioLDM baseline model and codebase
- Dataset contributors and TAs for verification

---

**Note**: This project is for academic/research purposes. The AudioLDM baseline remains unchanged and frozen as per project requirements.

