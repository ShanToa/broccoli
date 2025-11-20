"""
Reference Image-Based Synthetic Data Generation
Uses reference disease images directly without text prompts by conditioning
the input image with the reference disease style.
"""

import os
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import torch
from diffusers import StableDiffusionInpaintPipeline
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


def blend_reference_disease(image, mask, reference_disease, blend_strength=0.5):
    """
    Blend reference disease image into the masked region of the base image.
    This creates a conditioning image that guides the diffusion process.
    """
    mask_array = np.array(mask)
    coords = np.where(mask_array > 0)
    
    if len(coords[0]) == 0:
        return image
    
    # Get mask bounding box
    min_y, max_y = coords[0].min(), coords[0].max()
    min_x, max_x = coords[1].min(), coords[1].max()
    mask_h, mask_w = max_y - min_y, max_x - min_x
    
    # Resize reference to fit mask region (maintain aspect ratio)
    ref_aspect = reference_disease.width / reference_disease.height
    mask_aspect = mask_w / mask_h
    
    if ref_aspect > mask_aspect:
        new_w = mask_w
        new_h = int(mask_w / ref_aspect)
    else:
        new_h = mask_h
        new_w = int(mask_h * ref_aspect)
    
    ref_resized = reference_disease.resize((new_w, new_h), Image.LANCZOS)
    
    # Create image copy
    image_array = np.array(image).copy()
    ref_array = np.array(ref_resized)
    
    # Center reference in mask region
    start_y = min_y + (mask_h - new_h) // 2
    start_x = min_x + (mask_w - new_w) // 2
    end_y = start_y + new_h
    end_x = start_x + new_w
    
    # Ensure bounds
    start_y = max(0, start_y)
    start_x = max(0, start_x)
    end_y = min(image_array.shape[0], end_y)
    end_x = min(image_array.shape[1], end_x)
    
    # Adjust if needed
    if end_y - start_y != ref_array.shape[0] or end_x - start_x != ref_array.shape[1]:
        ref_array = np.array(Image.fromarray(ref_array).resize(
            (end_x - start_x, end_y - start_y), Image.LANCZOS))
    
    # Get region mask
    region_mask = mask_array[start_y:end_y, start_x:end_x]
    region_mask_norm = (region_mask > 0).astype(np.float32)[:, :, np.newaxis]
    
    # Blend reference into the region
    if region_mask_norm.shape[:2] == ref_array.shape[:2]:
        for c in range(3):
            original_region = image_array[start_y:end_y, start_x:end_x, c].astype(np.float32)
            ref_region = ref_array[:, :, c].astype(np.float32)
            
            blended = (
                (1 - blend_strength) * original_region +
                blend_strength * ref_region * region_mask_norm[:, :, 0]
            )
            image_array[start_y:end_y, start_x:end_x, c] = np.clip(blended, 0, 255).astype(np.uint8)
    
    return Image.fromarray(image_array)


def generate_with_reference(
    pipeline,
    base_image,
    mask,
    reference_disease_image,
    num_inference_steps=50,
    strength=0.85,
    guidance_scale=7.5,  # Higher for better quality, but still influenced by reference
    blend_strength=0.5,
    use_better_scheduler=True
):
    """
    Generate synthetic disease using reference image conditioning.
    Uses reference image blended into input to guide generation.
    Preserves original image resolution.
    """
    
    # Store original dimensions
    original_size = base_image.size  # (width, height)
    
    # Blend reference disease into the image to condition the model
    conditioned_image = blend_reference_disease(
        base_image, mask, reference_disease_image, blend_strength=blend_strength
    )
    
    # Use a descriptive prompt that matches the reference style
    # The reference image provides the visual style, prompt provides semantic guidance
    prompt = "disease spot on broccoli, brown lesion, natural lighting, high quality, detailed"
    negative_prompt = "blurry, distorted, low quality, artifacts, unrealistic, cartoon, painting"
    
    # Generate at original resolution
    # Stable Diffusion requires dimensions to be multiples of 8
    target_height = original_size[1]
    target_width = original_size[0]
    
    # Ensure dimensions are multiples of 8 (required by SD)
    # Round down to nearest multiple of 8 to avoid issues
    target_height = (target_height // 8) * 8
    target_width = (target_width // 8) * 8
    
    # Generate at the target resolution
    # Note: This will use more VRAM for larger images
    try:
        result = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=conditioned_image,
            mask_image=mask,
            height=target_height,
            width=target_width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=strength,
        ).images[0]
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"    Warning: Out of memory at {target_width}x{target_height}. Trying 1024x1024...")
            # Fallback to 1024x1024 if OOM
            result = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=conditioned_image,
                mask_image=mask,
                height=1024,
                width=1024,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                strength=strength,
            ).images[0]
            # Resize to original
            result = result.resize(original_size, Image.LANCZOS)
        else:
            raise
    
    # Resize back to exact original size if needed (in case of rounding or fallback)
    if result.size != original_size:
        result = result.resize(original_size, Image.LANCZOS)
    
    return result


def process_dataset(
    data_dir,
    output_dir,
    reference_disease_dir,
    model_id="runwayml/stable-diffusion-inpainting",
    num_images_per_sample=3,
    num_inference_steps=50,
    strength=0.85,
    guidance_scale=7.5,
    blend_strength=0.5,
    device="cuda" if torch.cuda.is_available() else "cpu",
    preserve_resolution=True
):
    """Process entire dataset to generate synthetic diseased images."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load reference disease images
    reference_images = load_reference_disease_images(reference_disease_dir)
    if not reference_images:
        print(f"Warning: No reference disease images found in {reference_disease_dir}")
        print("Please provide reference disease images in the reference_disease_dir")
        print("\nYou can extract disease regions from your data using:")
        print("  python -c \"from utils import extract_disease_regions; extract_disease_regions('data', 'reference_diseases')\"")
        return
    
    print(f"Found {len(reference_images)} reference disease images")
    
    # Load pipeline
    print(f"Loading Stable Diffusion model: {model_id}")
    print("This may take a few minutes on first run (downloading model)...")
    
    try:
        pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        pipeline = pipeline.to(device)
        
        # Use better scheduler for improved quality
        try:
            from diffusers import DPMSolverMultistepScheduler
            pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
            print("Using DPMSolverMultistepScheduler for better quality")
        except:
            print("Using default scheduler")
        
        # Enable progress bar for CPU mode so user can see it's working
        if device == "cpu":
            pipeline.set_progress_bar_config(disable=False)
            print("Note: CPU mode is very slow. Each image may take 5-10+ minutes.")
            # Reduce steps for CPU to make it faster
            if num_inference_steps > 30:
                print(f"Reducing inference steps from {num_inference_steps} to 30 for CPU speed")
                num_inference_steps = 30
        else:
            pipeline.set_progress_bar_config(disable=True)
            print(f"Using {num_inference_steps} inference steps for high quality generation")
        
        print(f"Model loaded. Using device: {device}")
        
        if device == "cuda" and preserve_resolution:
            print("Note: Generating at original resolution (1280x1280) will use more VRAM.")
            print("      If you get out-of-memory errors, consider using lower resolution.")
    except Exception as e:
        print(f"\nError loading model: {e}")
        print("\nTroubleshooting:")
        print("1. Check your internet connection (model needs to download)")
        print("2. Try: pip install --upgrade diffusers transformers")
        print("3. Clear cache: rm -r ~/.cache/huggingface/hub")
        raise
    
    # Get all image files
    image_files = list(Path(data_dir).glob("*.png")) + list(Path(data_dir).glob("*.jpg"))
    json_files = {f.stem: f for f in Path(data_dir).glob("*.json")}
    
    processed = 0
    skipped = 0
    
    print(f"\nProcessing {len(image_files)} images...")
    
    for img_file in tqdm(image_files, desc="Generating synthetic data"):
        img_name = img_file.stem
        
        if img_name not in json_files:
            skipped += 1
            continue
        
        try:
            image = Image.open(img_file).convert("RGB")
            mask, annotation_data = load_json_annotation(json_files[img_name])
            
            # Generate multiple variations
            for i in range(num_images_per_sample):
                # Randomly select a reference disease image
                ref_disease_path = random.choice(reference_images)
                ref_disease = Image.open(ref_disease_path).convert("RGB")
                
                # Generate synthetic disease
                synthetic_image = generate_with_reference(
                    pipeline=pipeline,
                    base_image=image,
                    mask=mask,
                    reference_disease_image=ref_disease,
                    num_inference_steps=num_inference_steps,
                    strength=strength,
                    guidance_scale=guidance_scale,
                    blend_strength=blend_strength,
                    use_better_scheduler=True
                )
                
                # Save output
                output_name = f"{img_name}_synthetic_{i+1}.png"
                output_path = os.path.join(output_dir, output_name)
                synthetic_image.save(output_path)
                
                # Save corresponding mask
                mask_output_path = os.path.join(output_dir, f"{img_name}_synthetic_{i+1}_mask.png")
                mask.save(mask_output_path)
                
                # Save annotation JSON
                annotation_output_path = os.path.join(output_dir, f"{img_name}_synthetic_{i+1}.json")
                annotation_data['imagePath'] = output_name
                with open(annotation_output_path, 'w') as f:
                    json.dump(annotation_data, f)
            
            processed += 1
            
        except Exception as e:
            print(f"\nError processing {img_file}: {e}")
            skipped += 1
            continue
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Processed: {processed} images")
    print(f"Skipped: {skipped} images")
    print(f"Generated: {processed * num_images_per_sample} synthetic images")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic broccoli disease data using reference images (no prompts)"
    )
    parser.add_argument("--data_dir", type=str, default="data",
                       help="Directory containing broccoli images and JSON annotations")
    parser.add_argument("--output_dir", type=str, default="synthetic_data",
                       help="Output directory for generated images")
    parser.add_argument("--reference_disease_dir", type=str, default="reference_diseases",
                       help="Directory containing reference disease images")
    parser.add_argument("--model_id", type=str, default="runwayml/stable-diffusion-inpainting",
                       help="Stable Diffusion model ID")
    parser.add_argument("--num_images", type=int, default=3,
                       help="Number of synthetic images to generate per input image")
    parser.add_argument("--num_inference_steps", type=int, default=75,
                       help="Number of inference steps (more = better quality but slower, 75 recommended for GPU)")
    parser.add_argument("--strength", type=float, default=0.85,
                       help="Strength of the transformation (0.0-1.0)")
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                       help="Guidance scale (higher = better quality, 7.5 recommended)")
    parser.add_argument("--blend_strength", type=float, default=0.5,
                       help="How much to blend reference image into conditioning (0.0-1.0, 0.5 recommended)")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use (cuda/cpu). Auto-detected if not specified")
    parser.add_argument("--no_preserve_resolution", action="store_true", default=False,
                       help="Don't preserve original resolution (use 512x512 instead). Default is to preserve original resolution (1280x1280)")
    
    args = parser.parse_args()
    
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device == "cpu":
        print("Warning: CPU mode will be very slow. GPU is highly recommended.")
    
    process_dataset(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        reference_disease_dir=args.reference_disease_dir,
        model_id=args.model_id,
        num_images_per_sample=args.num_images,
        num_inference_steps=args.num_inference_steps,
        strength=args.strength,
        guidance_scale=args.guidance_scale,
        blend_strength=args.blend_strength,
        device=device,
        preserve_resolution=not args.no_preserve_resolution
    )


if __name__ == "__main__":
    main()

