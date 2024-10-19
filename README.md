# Flux-Text-to-Image

**FLUX.1 [dev]** is a 12 billion parameter rectified flow transformer capable of generating images from text descriptions using diffusion-based models. This repository provides a guide to using the Flux model, which is built by Black Forest Labs and integrates with Hugging Face's `diffusers` library for ease of use.

## Table of Contents
- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Architecture](#architecture)
- [Theory](#theory)
- [Requirements](#requirements)
- [Acknowledgments](#acknowledgments)
- [License](#license)

## Introduction
Flux-Text-to-Image is an advanced text-to-image generation model that uses the FLUX.1 model from Black Forest Labs. It leverages a diffusion-based approach to gradually generate images from random noise guided by text prompts. This repository demonstrates how to use the model with minimal setup and seamless integration using the Hugging Face `diffusers` library.

## Prerequisites
Before getting started, ensure the following libraries are installed:
- `diffusers`
- `torch`
- `huggingface_hub`

You also need access to a GPU for faster and more efficient image generation.

## Installation
To install the required libraries, use the following command:
```bash
pip install -U diffusers torch huggingface_hub
```

Alternatively, you can install the dependencies from the provided `requirements.txt` file:
```bash
pip install -r requirements.txt
```

### `requirements.txt`
```txt
diffusers==0.12.0
torch==2.0.1
huggingface_hub==0.15.1
```

## Usage

### 1. **Import Libraries:**
    import torch
    from diffusers import FluxPipeline, DPMSolverMultistepScheduler
    from huggingface_hub import notebook_login


### 2. **Login to Hugging Face:**
    notebook_login()


### 3. **Load the Model:**

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16,
        revision="main",
        scheduler=DPMSolverMultistepScheduler.from_pretrained(
            "black-forest-labs/FLUX.1-dev", subfolder="scheduler"
        ),
    ).to(device)


### 4. **Define the Prompt and Generator:**

    prompt = "A cat holding a sign that says Hello India"
    generator = torch.Generator('cuda').manual_seed(0)


### 5. **Generate the Image:**

    image = pipe(
        prompt,
        height=512,
        width=512,
        guidance_scale=2,
        num_inference_steps=25,
        max_sequence_length=512,
        generator=generator
    ).images[0]


### 6. **Display and Save the Image:**

    from IPython.display import display
    display(image)
    image.save("flux-dev.png")


## Architecture
Flux-Text-to-Image employs a **rectified flow transformer architecture** and a **diffusion model** to generate images from textual descriptions. Here is an overview of the architecture and its key components:

### 1. **Text Encoder:**
   - The model uses a transformer-based text encoder (like CLIP or BERT) that converts the input text prompt into a latent space representation. 
   - The encoder captures contextual relationships between words and phrases to build a robust textual understanding.

### 2. **Image Decoder:**
   - The image decoder consists of a series of UNet-like structures, typical in diffusion models, responsible for generating images by iteratively refining noisy inputs.
   - The text latent representation conditions the image generation process, ensuring that the generated image aligns with the text.

### 3. **Diffusion Process:**
   - The model begins by generating a random noise image and iteratively refines it using a **denoising diffusion probabilistic model (DDPM)**, driven by the text prompt.
   - Each iteration, or inference step, adds clarity and structure to the image, gradually transforming it to match the text description.

### 4. **Scheduler:**
   - **DPMSolverMultistepScheduler** is employed to control the denoising process. It efficiently guides the generation, balancing between speed and quality.
   - The scheduler optimizes the noise reduction steps, ensuring the final output is both high-quality and semantically aligned with the input prompt.

### 5. **Guidance Mechanism:**
   - The model includes a **guidance scale**, which helps control how strictly the model adheres to the text input. A higher guidance scale leads to more text-aligned but potentially less creative outputs, while a lower scale increases diversity at the risk of deviating from the prompt.

## Theory
The Flux-Text-to-Image model is built upon diffusion-based image synthesis techniques. Key concepts include:

1. **Diffusion Process:** 
    - Starts with random noise and iteratively denoises the image based on a sequence of probabilistic refinements. Each step brings the image closer to a meaningful visual representation that aligns with the textual input.

2. **Guidance Scale:** 
    - A crucial hyperparameter that controls the strength of adherence to the text prompt. Higher values make the generated image more literal to the description, while lower values allow for more creative deviations.

3. **Multimodal Learning:**
    - The model integrates textual and visual information through a shared latent space, allowing the model to "understand" text descriptions in a visual context and generate corresponding images.

4. **Inference Steps:** 
    - The model generates images over multiple inference steps, balancing between output quality and computational efficiency. A higher number of steps leads to better quality but requires more processing time.

## Requirements
The following libraries are necessary for running the model. They can be installed directly using `requirements.txt`:

```txt
diffusers==0.12.0
torch==2.0.1
huggingface_hub==0.15.1
```

You will also need a GPU with sufficient VRAM (at least 12GB recommended) for efficient image generation.

## Acknowledgments
Special thanks to the team at Black Forest Labs for developing the FLUX.1 model, and to Hugging Face for providing the `diffusers` library and cloud infrastructure that make state-of-the-art AI models more accessible. This project would not have been possible without the contributions of these teams to open-source AI research.

## License
This project is licensed under the MIT License. For more details, refer to the [LICENSE](LICENSE) file.

