# Synthetic Broccoli Disease Data Generation

This project uses Stable Diffusion to generate synthetic broccoli disease images based on reference disease images (without text prompts).

## Features

- **Reference Image-Based Generation**: Uses reference disease images instead of text prompts
- **Mask-Based Inpainting**: Applies diseases to specific regions using masks from annotations
- **Batch Processing**: Processes entire datasets automatically
- **Multiple Variations**: Generates multiple synthetic variations per input image

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Prepare your data:
   - Place broccoli images and JSON annotation files in the `data/` directory
   - Create a `reference_diseases/` directory and add reference disease images

3. For IP-Adapter (optional, better reference image conditioning):
   - Download IP-Adapter model weights
   - Place in project root or specify path

## Usage

### Recommended: Reference-Based Generation (No Prompts)

This is the main script that uses reference images directly without text prompts:

```bash
python generate_reference_based.py \
    --data_dir data \
    --output_dir synthetic_data \
    --reference_disease_dir reference_diseases \
    --num_images 3 \
    --guidance_scale 1.5 \
    --blend_strength 0.4
```

### Alternative: Standard Inpainting

```bash
python generate_synthetic_data.py \
    --data_dir data \
    --output_dir synthetic_data \
    --reference_disease_dir reference_diseases \
    --num_images 3
```

### With IP-Adapter (Advanced)

```bash
python generate_with_ip_adapter.py \
    --data_dir data \
    --output_dir synthetic_data \
    --reference_disease_dir reference_diseases \
    --ip_adapter_model ip-adapter_sd15.bin \
    --num_images 3 \
    --ip_scale 1.0
```

### Parameters

- `--data_dir`: Directory containing input images and JSON annotations
- `--output_dir`: Directory to save generated synthetic images
- `--reference_disease_dir`: Directory containing reference disease images
- `--num_images`: Number of synthetic images to generate per input (default: 3)
- `--num_inference_steps`: Number of diffusion steps (default: 50)
- `--guidance_scale`: Guidance scale (default: 7.5)
- `--strength`: Strength of transformation (default: 0.8)
- `--device`: Device to use (cuda/cpu, auto-detected if not specified)

### Utility Scripts

Extract disease regions from existing images:
```bash
python -c "from utils import extract_disease_regions; extract_disease_regions('data', 'reference_diseases')"
```

## How It Works

The reference-based approach (`generate_reference_based.py`) works as follows:

1. **Load Data**: Reads broccoli images and their mask annotations from JSON files
2. **Reference Selection**: Randomly selects a reference disease image for each generation
3. **Image Conditioning**: Blends the reference disease image into the masked region of the base image
4. **Low-Guidance Generation**: Uses Stable Diffusion inpainting with very low guidance scale (1.5) so the model relies on the reference image rather than text prompts
5. **Output**: Saves synthetic images, masks, and annotations

The key is using a low `guidance_scale` (1.5) which minimizes the influence of text prompts and maximizes the influence of the reference image that's been blended into the input.

## Reference Disease Images

Place reference disease images (cropped disease spots) in the `reference_diseases/` directory. These images will be used as style references for generating synthetic diseases.

You can extract disease regions from your existing data using the utility function:
```python
from utils import extract_disease_regions
extract_disease_regions('data', 'reference_diseases')
```

## Output Structure

Generated files follow this naming:
- `{original_name}_synthetic_{n}.png` - Generated image
- `{original_name}_synthetic_{n}_mask.png` - Corresponding mask
- `{original_name}_synthetic_{n}.json` - Annotation file

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- ~10GB+ VRAM for GPU usage
- Stable Diffusion model weights (downloaded automatically)

## Notes

- First run will download the Stable Diffusion model (~4GB)
- GPU is highly recommended for faster processing
- Adjust `strength` parameter to control how much the reference image influences the output
- For IP-Adapter, higher `ip_scale` values make the reference image more influential

