import os

import torch

api_key = os.environ.get("OPENROUTER_API_KEY", "")
model = os.environ.get("OPENROUTER_MODEL", "google/gemini-2.5-flash")
openai_api_key = os.environ.get("OPENAI_API_KEY", "")

device = torch.device(os.environ.get("TORCH_DEVICE", "cuda:1" if torch.cuda.is_available() else "cpu"))

IMAGE_HEIGHT = 1024
IMAGE_WIDTH = 1024
