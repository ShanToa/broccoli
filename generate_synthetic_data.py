"""
Synthetic Data Generation for Broccoli Disease Spots using Stable Diffusion
This script generates synthetic diseased broccoli images using reference disease images
instead of text prompts.
"""

import os
import json
import numpy as np
from PIL import Image, ImageDraw
import torch
from diffusers import StableDiffusionInpaintPipeline, StableDiffusionImg2ImgPipeline
from diffusers import ControlNetModel, StableDiffusionControlNetInpaintPipeline
from diffusers.utils import load_image
import argparse
from pathlib import Path
from tqdm import tqdm
import random


def load_json_annotation(json_path):
    """Load polygon annotations from JSON file and create mask."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Get image dimensions
    height = data.get('imageHeight', 1280)
    width = data.get('imageWidth', 1280)
    
    # Create mask
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)
    
    # Draw all polygons
    for shape in data.get('shapes', []):
        if shape.get('shape_type') == 'polygon':
            points = shape.get('points', [])
            if len(points) >= 3:
                # Convert to tuple format
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


def extract_disease_patch(image, mask, bbox_padding=10):
    """Extract disease patch from image using mask bounding box."""
    mask_array = np.array(mask)
    coords = np.where(mask_array > 0)
    
    if len(coords[0]) == 0:
        return None
    
    min_y, max_y = max(0, coords[0].min() - bbox_padding), min(mask_array.shape[0], coords[0].max() + bbox_padding)
    min_x, max_x = max(0, coords[1].min() - bbox_padding), min(mask_array.shape[1], coords[1].max() + bbox_padding)
    
    patch = image.crop((min_x, min_y, max_x, max_y))
    mask_patch = mask.crop((min_x, min_y, max_x, max_y))
    
    return patch, mask_patch, (min_x, min_y, max_x, max_y)


def prepare_inpaint_inputs(image, mask, reference_disease, blend_strength=0.4):
    """Prepare inputs for inpainting with reference disease style."""
    # Resize reference disease to match mask size approximately
    mask_array = np.array(mask)
    coords = np.where(mask_array > 0)
    
    if len(coords[0]) == 0:
        return image, mask
    
    # Get mask dimensions
    min_y, max_y = coords[0].min(), coords[0].max()
    min_x, max_x = coords[1].min(), coords[1].max()
    mask_h, mask_w = max_y - min_y, max_x - min_x
    
    # Resize reference disease to match mask region (maintain aspect ratio if needed)
    ref_aspect = reference_disease.width / reference_disease.height
    mask_aspect = mask_w / mask_h
    
    if ref_aspect > mask_aspect:
        # Reference is wider, fit to width
        new_w = mask_w
        new_h = int(mask_w / ref_aspect)
    else:
        # Reference is taller, fit to height
        new_h = mask_h
        new_w = int(mask_h * ref_aspect)
    
    ref_resized = reference_disease.resize((new_w, new_h), Image.LANCZOS)
    
    # Create a blended image for conditioning
    image_array = np.array(image).copy()
    mask_binary = (mask_array > 0).astype(np.float32)[:, :, np.newaxis]
    
    # Blend reference disease into the masked region
    ref_array = np.array(ref_resized)
    
    # Center the reference in the mask region
    start_y = min_y + (mask_h - new_h) // 2
    start_x = min_x + (mask_w - new_w) // 2
    end_y = start_y + new_h
    end_x = start_x + new_w
    
    # Ensure we don't go out of bounds
    start_y = max(0, start_y)
    start_x = max(0, start_x)
    end_y = min(image_array.shape[0], end_y)
    end_x = min(image_array.shape[1], end_x)
    
    # Adjust ref_array if needed
    if end_y - start_y != ref_array.shape[0] or end_x - start_x != ref_array.shape[1]:
        ref_array = np.array(Image.fromarray(ref_array).resize(
            (end_x - start_x, end_y - start_y), Image.LANCZOS))
    
    # Blend reference disease into the masked region
    region_mask = mask_binary[start_y:end_y, start_x:end_x]
    if region_mask.shape[:2] == ref_array.shape[:2]:
        for c in range(3):
            image_array[start_y:end_y, start_x:end_x, c] = (
                (1 - blend_strength) * image_array[start_y:end_y, start_x:end_x, c] +
                blend_strength * ref_array[:, :, c] * region_mask[:, :, 0]
            )
    
    blended_image = Image.fromarray(image_array.astype(np.uint8))
    
    return blended_image, mask


def generate_synthetic_disease(
    pipeline,
    base_image,
    mask,
    reference_disease_image,
    num_inference_steps=50,
    guidance_scale=7.5,
    strength=0.8,
    use_inpaint=True
):
    """Generate synthetic disease using Stable Diffusion with reference image."""
    
    # Prepare the image and mask by blending reference disease
    # This conditions the model on the reference image style
    prepared_image, prepared_mask = prepare_inpaint_inputs(
        base_image, mask, reference_disease_image, blend_strength=0.3
    )
    
    if use_inpaint:
        # Use inpainting pipeline - this works better for mask-based generation
        # Using a minimal prompt that describes the reference style
        result = pipeline(
            prompt="disease spot, brown spot, lesion",  # Minimal prompt for guidance
            negative_prompt="blurry, distorted, low quality",
            image=prepared_image,
            mask_image=prepared_mask,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=strength,
        ).images[0]
    else:
        # Use img2img pipeline as fallback
        result = pipeline(
            prompt="disease spot",
            image=prepared_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=strength,
        ).images[0]
    
    return result


def process_dataset(
    data_dir,
    output_dir,
    reference_disease_dir,
    model_id="runwayml/stable-diffusion-inpainting",
    num_images_per_sample=3,
    num_inference_steps=50,
    guidance_scale=7.5,
    strength=0.8,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    """Process entire dataset to generate synthetic diseased images."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load reference disease images
    reference_images = load_reference_disease_images(reference_disease_dir)
    if not reference_images:
        print(f"Warning: No reference disease images found in {reference_disease_dir}")
        print("Please provide reference disease images in the reference_disease_dir")
        return
    
    print(f"Found {len(reference_images)} reference disease images")
    
    # Load pipeline
    print(f"Loading Stable Diffusion model: {model_id}")
    if "inpaint" in model_id.lower():
        pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        use_inpaint = True
    else:
        pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        use_inpaint = False
    
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)
    
    # Get all image files
    image_files = list(Path(data_dir).glob("*.png")) + list(Path(data_dir).glob("*.jpg"))
    json_files = {f.stem: f for f in Path(data_dir).glob("*.json")}
    
    processed = 0
    skipped = 0
    
    for img_file in tqdm(image_files, desc="Processing images"):
        img_name = img_file.stem
        
        # Check if corresponding JSON exists
        if img_name not in json_files:
            skipped += 1
            continue
        
        # Load image and mask
        try:
            image = Image.open(img_file).convert("RGB")
            mask, annotation_data = load_json_annotation(json_files[img_name])
            
            # Generate multiple variations
            for i in range(num_images_per_sample):
                # Randomly select a reference disease image
                ref_disease_path = random.choice(reference_images)
                ref_disease = Image.open(ref_disease_path).convert("RGB")
                
                # Generate synthetic disease
                synthetic_image = generate_synthetic_disease(
                    pipeline=pipeline,
                    base_image=image,
                    mask=mask,
                    reference_disease_image=ref_disease,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    strength=strength,
                    use_inpaint=use_inpaint
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
            print(f"Error processing {img_file}: {e}")
            skipped += 1
            continue
    
    print(f"\nProcessing complete!")
    print(f"Processed: {processed} images")
    print(f"Skipped: {skipped} images")
    print(f"Generated: {processed * num_images_per_sample} synthetic images")
    print(f"Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic broccoli disease data using Stable Diffusion")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing broccoli images and JSON annotations")
    parser.add_argument("--output_dir", type=str, default="synthetic_data", help="Output directory for generated images")
    parser.add_argument("--reference_disease_dir", type=str, default="reference_diseases", 
                       help="Directory containing reference disease images")
    parser.add_argument("--model_id", type=str, default="runwayml/stable-diffusion-inpainting",
                       help="Stable Diffusion model ID")
    parser.add_argument("--num_images", type=int, default=3, 
                       help="Number of synthetic images to generate per input image")
    parser.add_argument("--num_inference_steps", type=int, default=50,
                       help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                       help="Guidance scale")
    parser.add_argument("--strength", type=float, default=0.8,
                       help="Strength of the transformation")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use (cuda/cpu). Auto-detected if not specified")
    
    args = parser.parse_args()
    
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    process_dataset(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        reference_disease_dir=args.reference_disease_dir,
        model_id=args.model_id,
        num_images_per_sample=args.num_images,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        strength=args.strength,
        device=device
    )


if __name__ == "__main__":
    main()

