"""
Utility functions for data processing and visualization
"""

import os
import json
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path


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


def visualize_mask_overlay(image_path, json_path, output_path=None):
    """Visualize image with mask overlay."""
    image = Image.open(image_path).convert("RGB")
    mask, _ = load_json_annotation(json_path)
    
    # Create overlay
    overlay = image.copy()
    mask_rgb = Image.new("RGB", mask.size)
    mask_rgb.paste(mask)
    
    # Blend
    blended = Image.blend(overlay, mask_rgb, 0.4)
    
    if output_path:
        blended.save(output_path)
    
    return blended


def extract_disease_regions(data_dir, output_dir):
    """Extract disease regions from images using masks."""
    os.makedirs(output_dir, exist_ok=True)
    
    image_files = list(Path(data_dir).glob("*.png"))
    json_files = {f.stem: f for f in Path(data_dir).glob("*.json")}
    
    extracted = []
    
    for img_file in image_files:
        img_name = img_file.stem
        if img_name not in json_files:
            continue
        
        try:
            image = Image.open(img_file).convert("RGB")
            mask, _ = load_json_annotation(json_files[img_name])
            
            # Get bounding box
            mask_array = np.array(mask)
            coords = np.where(mask_array > 0)
            
            if len(coords[0]) == 0:
                continue
            
            min_y, max_y = coords[0].min(), coords[0].max()
            min_x, max_x = coords[1].min(), coords[1].max()
            
            # Extract region
            region = image.crop((min_x, min_y, max_x, max_y))
            region_mask = mask.crop((min_x, min_y, max_x, max_y))
            
            # Apply mask to region
            region_array = np.array(region)
            mask_region_array = np.array(region_mask)
            region_array[mask_region_array == 0] = [255, 255, 255]  # White background
            
            output_image = Image.fromarray(region_array)
            output_path = os.path.join(output_dir, f"{img_name}_disease_region.png")
            output_image.save(output_path)
            extracted.append(output_path)
            
        except Exception as e:
            print(f"Error extracting from {img_file}: {e}")
            continue
    
    print(f"Extracted {len(extracted)} disease regions to {output_dir}")
    return extracted

