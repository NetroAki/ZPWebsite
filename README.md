# Chicken Image Generator

This script generates images of chickens using Stable Diffusion, a state-of-the-art text-to-image generation model.

## Features

- Generate realistic chicken images from text descriptions
- Multiple preset chicken prompts to choose from
- GPU acceleration support for faster generation
- High-quality image output

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the script to generate a chicken image:
```bash
python generate_chicken_image.py
```

The script will:
1. Download the Stable Diffusion model (first run only)
2. Generate an image based on the prompt
3. Save the result as `chicken_image.png`

## Customization

You can modify the script to:
- Change the prompt by editing the `chicken_prompts` list
- Adjust generation parameters like `num_inference_steps` and `guidance_scale`
- Change the output filename

## Example Prompts

The script includes several chicken prompts:
- "A realistic chicken standing in a farmyard"
- "A beautiful brown chicken with detailed feathers"
- "A white chicken in a green field"
- "A cartoon chicken with a friendly expression"
- "A detailed portrait of a rooster with colorful feathers"

## Requirements

- Python 3.7+
- CUDA-compatible GPU (optional, for faster generation)
- Internet connection (for model download)

## Output

The generated image will be saved as `chicken_image.png` in the current directory.