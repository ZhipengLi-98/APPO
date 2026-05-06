# Human-in-the-Loop Image Prompt Optimization App

This Flask app runs an interactive text-to-image prompt optimization workflow. A user enters an initial prompt, reviews generated images, selects preferred outputs, and the app uses those choices to produce improved prompt variants for the next iteration.

## Setup

Create and activate a Python environment using your preferred tool, then install the dependencies:

```bash
pip install -r requirements.txt
```

The app loads Stable Diffusion XL from Hugging Face through `diffusers`, so the first run may download model weights. A CUDA-capable GPU is recommended.

## Configuration

Set the language-model API key before starting the app:

```bash
export OPENROUTER_API_KEY="your-openrouter-key"
```

Optional settings:

```bash
export OPENROUTER_MODEL="google/gemini-2.5-flash"
export TORCH_DEVICE="cuda:0"
export FLASK_SECRET_KEY="replace-this-for-shared-deployments"
export PORT=5002
export USER_ID=0
export CONDITION_ID=0
export TASK_ID=0
```

You can also copy `.env.example` as a starting point if your deployment workflow loads environment files.

## Run

From this folder:

```bash
python app.py
```

Then open:

```text
http://localhost:5002
```

Generated study outputs are written under:

```text
static/generated/
```

## Notes

- Keep API keys in environment variables. Do not commit them to the repository.
- Generated images and study outputs are intentionally ignored by Git.
- The app is intended for research and user-study use rather than production deployment as-is.
