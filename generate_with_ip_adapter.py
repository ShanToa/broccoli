"""
Synthetic Data Generation using Stable Diffusion with IP-Adapter
This version uses IP-Adapter to condition on reference disease images without prompts.
"""

import os
import json
import numpy as np
from PIL import Image, ImageDraw
import torch
from diffusers import StableDiffusionInpaintPipeline
from diffusers import AutoencoderKL
from transformers import CLIPImageProcessor
from ip_adapter import IPAdapter
import argparse
from pathlib import Path
from tqdm import tqdm
import random


def load_json_annotation(json_path):
    """Load polygon annotations from JSON file and create mask."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    height = data.get('imageHeight', 1280)
    width = data.get('imageWidth', 1280)
    
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)
    
    for shape in data.get('shapes', []):
        if shape.get('shape_type') == 'polygon':
            points = shape.get('points', [])
            if len(points) >= 3:
                polygon_points = [(int(p[0]), int(p[1])) for p in points]
                draw.polygon(polygon_points, fill=255)
    
    return mask, data


def load_reference_disease_images(reference_dir):
    """Load reference disease images from directory."""
    reference_images = []
    if os.path.exists(reference_dir):
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            reference_images.extend(Path(reference_dir).glob(ext))
            reference_images.extend(Path(reference_dir).glob(ext.upper()))
    return [str(img) for img in reference_images]


def generate_with_ip_adapter(
    pipeline,
    ip_adapter,
    base_image,
    mask,
    reference_disease_image,
    num_inference_steps=50,
    guidance_scale=7.5,
    strength=0.8,
    ip_scale=1.0
):
    """Generate synthetic disease using IP-Adapter with reference image."""
    
    # Prepare reference image
    ref_image = reference_disease_image.convert("RGB")
    
    # Generate using IP-Adapter
    result = pipeline(
        prompt="",  # Empty prompt - using only reference image
        negative_prompt="blurry, low quality, distorted",
        image=base_image,
        mask_image=mask,
        ip_adapter_image=ref_image,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        strength=strength,
        ip_scale=ip_scale,
    ).images[0]
    
    return result


def process_dataset(
    data_dir,
    output_dir,
    reference_disease_dir,
    base_model_id="runwayml/stable-diffusion-inpainting",
    ip_adapter_model_path="ip-adapter_sd15.bin",
    num_images_per_sample=3,
    num_inference_steps=50,
    guidance_scale=7.5,
    strength=0.8,
    ip_scale=1.0,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    """Process entire dataset to generate synthetic diseased images."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load reference disease images
    reference_images = load_reference_disease_images(reference_disease_dir)
    if not reference_images:
        print(f"Warning: No reference disease images found in {reference_disease_dir}")
        print("Please provide reference disease images in the reference_disease_dir")
        return
    
    print(f"Found {len(reference_images)} reference disease images")
    
    # Load base pipeline
    print(f"Loading Stable Diffusion model: {base_model_id}")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
        requires_safety_checker=False
    )
    pipe = pipe.to(device)
    
    # Load IP-Adapter
    print(f"Loading IP-Adapter from: {ip_adapter_model_path}")
    try:
        ip_adapter = IPAdapter(pipe, ip_adapter_model_path, device, num_tokens=16)
    except Exception as e:
        print(f"Error loading IP-Adapter: {e}")
        print("Falling back to standard inpainting without IP-Adapter")
        ip_adapter = None
    
    pipe.set_progress_bar_config(disable=True)
    
    # Get all image files
    image_files = list(Path(data_dir).glob("*.png")) + list(Path(data_dir).glob("*.jpg"))
    json_files = {f.stem: f for f in Path(data_dir).glob("*.json")}
    
    processed = 0
    skipped = 0
    
    for img_file in tqdm(image_files, desc="Processing images"):
        img_name = img_file.stem
        
        if img_name not in json_files:
            skipped += 1
            continue
        
        try:
            image = Image.open(img_file).convert("RGB")
            mask, annotation_data = load_json_annotation(json_files[img_name])
            
            for i in range(num_images_per_sample):
                ref_disease_path = random.choice(reference_images)
                ref_disease = Image.open(ref_disease_path).convert("RGB")
                
                if ip_adapter:
                    synthetic_image = generate_with_ip_adapter(
                        pipeline=pipe,
                        ip_adapter=ip_adapter,
                        base_image=image,
                        mask=mask,
                        reference_disease_image=ref_disease,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        strength=strength,
                        ip_scale=ip_scale
                    )
                else:
                    # Fallback to standard inpainting
                    synthetic_image = pipe(
                        prompt="disease spot on broccoli",
                        image=image,
                        mask_image=mask,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        strength=strength,
                    ).images[0]
                
                # Save output
                output_name = f"{img_name}_synthetic_{i+1}.png"
                output_path = os.path.join(output_dir, output_name)
                synthetic_image.save(output_path)
                
                mask_output_path = os.path.join(output_dir, f"{img_name}_synthetic_{i+1}_mask.png")
                mask.save(mask_output_path)
                
                annotation_output_path = os.path.join(output_dir, f"{img_name}_synthetic_{i+1}.json")
                annotation_data['imagePath'] = output_name
                with open(annotation_output_path, 'w') as f:
                    json.dump(annotation_data, f)
            
            processed += 1
            
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
            skipped += 1
            continue
    
    print(f"\nProcessing complete!")
    print(f"Processed: {processed} images")
    print(f"Skipped: {skipped} images")
    print(f"Generated: {processed * num_images_per_sample} synthetic images")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic data with IP-Adapter")
    parser.add_argument("--data_dir", type=str, default="data", help="Input data directory")
    parser.add_argument("--output_dir", type=str, default="synthetic_data", help="Output directory")
    parser.add_argument("--reference_disease_dir", type=str, default="reference_diseases",
                       help="Directory with reference disease images")
    parser.add_argument("--base_model_id", type=str, default="runwayml/stable-diffusion-inpainting",
                       help="Base Stable Diffusion model")
    parser.add_argument("--ip_adapter_model", type=str, default="ip-adapter_sd15.bin",
                       help="IP-Adapter model path")
    parser.add_argument("--num_images", type=int, default=3, help="Images per sample")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--strength", type=float, default=0.8)
    parser.add_argument("--ip_scale", type=float, default=1.0, help="IP-Adapter scale")
    parser.add_argument("--device", type=str, default=None)
    
    args = parser.parse_args()
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    
    process_dataset(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        reference_disease_dir=args.reference_disease_dir,
        base_model_id=args.base_model_id,
        ip_adapter_model_path=args.ip_adapter_model,
        num_images_per_sample=args.num_images,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        strength=args.strength,
        ip_scale=args.ip_scale,
        device=device
    )


if __name__ == "__main__":
    main()

