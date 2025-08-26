#!/usr/bin/env python3
"""
Generate an image of a chicken using text-to-image generation.
"""

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import os

def generate_chicken_image(prompt="A realistic chicken standing in a farmyard", 
                          output_path="chicken_image.png",
                          num_inference_steps=50,
                          guidance_scale=7.5):
    """
    Generate an image of a chicken using Stable Diffusion.
    
    Args:
        prompt (str): Text description of the image to generate
        output_path (str): Path to save the generated image
        num_inference_steps (int): Number of denoising steps
        guidance_scale (float): Guidance scale for classifier-free guidance
    """
    
    print("Loading Stable Diffusion model...")
    
    # Load the pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        use_safetensors=True
    )
    
    # Move to GPU if available
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
        print("Using GPU for generation")
    else:
        print("Using CPU for generation (this will be slower)")
    
    print(f"Generating image with prompt: '{prompt}'")
    
    # Generate the image
    image = pipe(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale
    ).images[0]
    
    # Save the image
    image.save(output_path)
    print(f"Image saved to: {output_path}")
    
    return image

def main():
    """Main function to generate chicken image."""
    
    # Different chicken prompts you can try
    chicken_prompts = [
        "A realistic chicken standing in a farmyard",
        "A beautiful brown chicken with detailed feathers",
        "A white chicken in a green field",
        "A cartoon chicken with a friendly expression",
        "A detailed portrait of a rooster with colorful feathers"
    ]
    
    print("Chicken Image Generator")
    print("=" * 30)
    
    # Use the first prompt by default
    prompt = chicken_prompts[0]
    
    try:
        # Generate the image
        image = generate_chicken_image(prompt=prompt)
        
        print("\nImage generation completed successfully!")
        print(f"Generated image saved as: chicken_image.png")
        
        # Display image info
        print(f"Image size: {image.size}")
        print(f"Image mode: {image.mode}")
        
    except Exception as e:
        print(f"Error generating image: {e}")
        print("\nMake sure you have the required dependencies installed:")
        print("pip install torch diffusers transformers accelerate")

if __name__ == "__main__":
    main()