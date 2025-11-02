import yaml
import os
import torch
import gradio as gr
from PIL import Image
import torchvision.transforms as T
from audioldm import build_model, save_wave
from audioldm.ciap.models.image_encoder import ImageEncoder
from audioldm.ciap.models.audio_encoder import AudioEncoder
from audioldm.ciap.models.ciap_cond import CIAPCondStage
import tempfile

# Global variables for loaded models
config = None
audioldm = None
cond = None
device = None
image_encoder = None
audio_encoder = None

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_mapped_state(model, ckpt_path, mapping_rules=None, device="cpu"):
    sd = torch.load(ckpt_path, map_location=device)
    if isinstance(sd, dict) and any(k in sd for k in ("state_dict", "model_state_dict")):
        for key in ("state_dict", "model_state_dict"):
            if key in sd:
                sd = sd[key]
                break

    model_keys = set(model.state_dict().keys())
    new_sd = {}
    skipped = []
    for k, v in list(sd.items()):
        nk = k
        if mapping_rules:
            for src, dst in mapping_rules.items():
                if nk.startswith(src):
                    nk = nk.replace(src, dst, 1)
        if nk not in model_keys and nk.startswith("proj."):
            nk = nk.replace("proj.", "fc.", 1)
        if nk in model_keys:
            new_sd[nk] = v
        else:
            skipped.append(k)

    load_res = model.load_state_dict(new_sd, strict=False)
    if skipped:
        print(f"Skipped {len(skipped)} keys from checkpoint")
    return load_res

def initialize_models():
    """Initialize all models (call once at startup)"""
    global config, audioldm, cond, device, image_encoder, audio_encoder
    
    if audioldm is not None:
        return  # Already initialized
    
    print("Initializing models...")
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load config
    config_path = os.path.join(script_dir, "audioldm", "ciap", "configs", "ciap_config.yaml")
    config = load_config(config_path)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load AudioLDM
    ldm_ckpt_path = os.path.join(script_dir, "ckpt", "audioldm-s-full.ckpt")
    audioldm = build_model(ckpt_path=ldm_ckpt_path)
    audioldm.eval()
    print("Loaded AudioLDM model")
    
    # Initialize CIAP encoders
    ckpt_dir = os.path.join(script_dir, "ckpt")
    image_ckpt = os.path.join(ckpt_dir, "ciap_image_encoder_epoch110.pt")
    audio_ckpt = os.path.join(ckpt_dir, "ciap_audio_encoder2.pt")
    
    image_encoder = ImageEncoder(config["model"]["image_encoder"]).to(device)
    audio_encoder = AudioEncoder(config["model"]["audio_encoder"]).to(device)
    
    # Load checkpoints
    try:
        mapping = {"proj.": "fc."}
        load_mapped_state(image_encoder, image_ckpt, mapping_rules=mapping, device=device)
        print("Loaded image encoder checkpoint")
    except Exception as e:
        print(f"Warning: Failed to load image encoder checkpoint: {e}")
        try:
            image_encoder.load_state_dict(torch.load(image_ckpt, map_location=device), strict=False)
        except Exception as e2:
            raise RuntimeError(f"Failed to load image checkpoint: {e2}")
    
    if os.path.exists(audio_ckpt):
        try:
            audio_encoder.load_state_dict(torch.load(audio_ckpt, map_location=device), strict=False)
            print("Loaded audio encoder checkpoint")
        except Exception as e:
            print(f"Warning: Failed to load audio encoder checkpoint: {e}")
    
    image_encoder.eval()
    audio_encoder.eval()
    
    # Build cond stage
    cond = CIAPCondStage(image_encoder, audio_encoder, embed_dim=512, device=device)
    
    print("All models initialized successfully!")

def preprocess_uploaded_image(image, target_size=(224, 224)):
    """
    Preprocess uploaded image to handle different formats and dimensions.
    Args:
        image: PIL Image or numpy array from Gradio
        target_size: Target size (H, W) - from config it's 224x224
    Returns:
        Preprocessed image tensor [1, C, H, W]
    """
    # Convert to PIL Image if needed
    if isinstance(image, Image.Image):
        pil_image = image
    else:
        # Gradio might return numpy array
        pil_image = Image.fromarray(image)
    
    # Convert to RGB (handles RGBA, L, etc.)
    pil_image = pil_image.convert("RGB")
    
    # Get original size
    orig_size = pil_image.size  # (W, H)
    print(f"Original image size: {orig_size}")
    
    # Create transform - match the dataset preprocessing (Resize + ToTensor only)
    # The dataset uses: T.Compose([T.Resize(image_size), T.ToTensor()])
    transform = T.Compose([
        T.Resize(target_size),  # (224, 224) from config
        T.ToTensor()  # Converts to [0, 1] range and [C, H, W] format
    ])
    
    # Apply transform
    img_tensor = transform(pil_image)  # [C, H, W]
    
    # Add batch dimension
    img_tensor = img_tensor.unsqueeze(0)  # [1, C, H, W]
    
    return img_tensor

def duration_to_latent_t_size(duration):
    """Convert duration in seconds to latent time size"""
    return int(duration * 25.6)

def generate_audio_from_image(image, desired_seconds=5, progress=gr.Progress()):
    """
    Generate audio from uploaded image.
    Args:
        image: Uploaded image (PIL Image or numpy array)
        desired_seconds: Desired audio duration in seconds
        progress: Gradio progress tracker
    Returns:
        Path to generated audio file
    """
    global audioldm, cond, device
    
    if image is None:
        raise gr.Error("Please upload an image first!")
    
    if audioldm is None:
        raise gr.Error("Models not initialized! Please wait...")
    
    try:
        progress(0.1, desc="Preprocessing image...")
        
        # Preprocess image
        image_tensor = preprocess_uploaded_image(image)
        image_tensor = image_tensor.to(device)
        
        progress(0.3, desc="Encoding image to embedding...")
        
        # Set latent_t_size based on duration
        audioldm.latent_t_size = duration_to_latent_t_size(desired_seconds)
        
        # Encode image to embedding
        with torch.no_grad():
            img_emb = cond.encode(image_tensor)  # [B, D]
            print(f"Image embedding shape: {img_emb.shape}")
            
            progress(0.5, desc="Generating audio waveform (this may take a while)...")
            
            # Sample latent from AudioLDM conditioned on image embedding
            # The sample method shows progress internally with tqdm
            waveform_latent = audioldm.sample(cond=img_emb, batch_size=img_emb.shape[0])
            
            progress(0.8, desc="Decoding waveform to audio...")
            
            # Decode to mel + waveform
            mel = audioldm.decode_first_stage(waveform_latent)
            waveform = audioldm.mel_spectrogram_to_waveform(mel)
            
            progress(0.9, desc="Post-processing audio...")
            
            # Crop to desired duration
            sample_rate = 16000
            target_len = desired_seconds * sample_rate
            if waveform.shape[-1] > target_len:
                waveform = waveform[..., :target_len]
            
            # Save to temporary file
            output_dir = tempfile.mkdtemp()
            
            # save_wave saves as "name_0.wav", "name_1.wav", etc.
            save_wave(waveform, savepath=output_dir, name="generated_audio")
            
            # Find the actual saved file (should be generated_audio_0.wav for first batch)
            actual_path = os.path.join(output_dir, "generated_audio_0.wav")
            
            if not os.path.exists(actual_path):
                # Fallback: look for any .wav file in the directory
                wav_files = [f for f in os.listdir(output_dir) if f.endswith('.wav')]
                if wav_files:
                    actual_path = os.path.join(output_dir, wav_files[0])
            
            progress(1.0, desc="Complete!")
            
            return actual_path
            
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error during generation: {error_details}")
        raise gr.Error(f"Error during generation: {str(e)}")

# Create Gradio interface
def create_ui():
    # Initialize models
    initialize_models()
    
    # Custom CSS for better UI
    css = """
    .gradio-container {
        font-family: 'Arial', sans-serif;
    }
    .main-header {
        text-align: center;
        margin-bottom: 20px;
    }
    """
    
    with gr.Blocks(css=css, title="Image-to-Audio Generation") as demo:
        gr.Markdown(
            """
            # 🎵 Image-to-Audio Generation
            Upload an image and generate audio that matches the scene!
            
            **Note:** The first generation may take longer as models load. Subsequent generations will be faster.
            """,
            elem_classes=["main-header"]
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    label="Upload Image",
                    type="pil",
                    height=300
                )
                
                duration_slider = gr.Slider(
                    minimum=2.5,
                    maximum=10.0,
                    value=5.0,
                    step=2.5,
                    label="Audio Duration (seconds)"
                )
                
                generate_btn = gr.Button(
                    "Generate Audio 🎵",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=1):
                audio_output = gr.Audio(
                    label="Generated Audio",
                    type="filepath",
                    autoplay=False
                )
                
                status_text = gr.Textbox(
                    label="Status",
                    value="Ready to generate! Upload an image and click 'Generate Audio'.",
                    interactive=False
                )
        
        # Examples section
        gr.Markdown("### Example Images")
        gr.Examples(
            examples=[],
            inputs=image_input
        )
        
        # Generation function with progress
        def generate_with_status(image, duration):
            try:
                status_text = "Generating audio... Please wait."
                audio_path = generate_audio_from_image(image, duration)
                status_text = "✅ Audio generated successfully! You can play it above."
                return audio_path, status_text
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                return None, error_msg
        
        generate_btn.click(
            fn=generate_with_status,
            inputs=[image_input, duration_slider],
            outputs=[audio_output, status_text]
        )
        
        gr.Markdown(
            """
            ---
            **How it works:**
            1. Upload an image (supports various formats: JPG, PNG, etc.)
            2. The image is automatically resized and preprocessed
            3. A diffusion model generates audio matching the image
            4. Play the generated audio directly in the browser
            """
        )
    
    return demo

if __name__ == "__main__":
    demo = create_ui()
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

