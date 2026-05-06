import torch
from diffusers import StableDiffusionXLPipeline
from diffusers.utils import load_image
from PIL import Image
import os
from utils import device, IMAGE_HEIGHT, IMAGE_WIDTH

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
).to(device)

def generate_image_online(prompt):
    image = pipe(prompt=prompt,
                height=IMAGE_HEIGHT,
                width=IMAGE_WIDTH, 
                num_inference_steps=50, 
                guidance_scale=7.5).images[0]
    return image

def generate_image_xl(prompt, output_dir):
    if prompt == "":
        return

    if type(prompt) == list:
        for i in range(len(prompt)):
            os.makedirs(output_dir, exist_ok=True)

            # Generate image
            image = pipe(prompt=prompt[i],
                        height=IMAGE_HEIGHT,
                        width=IMAGE_WIDTH, 
                        num_inference_steps=50, 
                        guidance_scale=7.5).images[0]

            print(f"{output_dir}/{i}.png")
            # Save image
            image.save(f"{output_dir}/{i}.png")

    else:
        # Create the folder only if it does not exist
        # os.makedirs(output_dir, exist_ok=True)

            # Generate image
        image = pipe(prompt=prompt,
                    height=IMAGE_HEIGHT,
                    width=IMAGE_WIDTH, 
                    num_inference_steps=50, 
                    guidance_scale=7.5).images[0]

        # Save image
        image.save(output_dir)

if __name__ == "__main__":
    # Load the SDXL base pipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    ).to(device)

    # Define your prompt
    prompt = "['crossroads intersection near', 'traffic light', 'waterway', 'signpost']"

    # Generate image
    image = pipe(prompt=prompt, num_inference_steps=30, guidance_scale=7.5).images[0]

    # Save image
    image.save("output_sdxl.png")
