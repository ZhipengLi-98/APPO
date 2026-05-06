# Human-in-the-Loop Image Prompt Optimization App

This is the final user-study app for the proposed method. It lets a user enter an image-generation prompt, choose preferred images across iterations, and uses those preferences to generate improved prompt variants.

## Environment

Run all commands in the `py39` conda environment:

```bash
conda activate py39
```

For non-interactive shells, use:

```bash
conda run -n py39 <command>
```

## Configuration

Set the language-model API key before starting the app:

```bash
export OPENROUTER_API_KEY="your-openrouter-key"
```

Optional settings:

```bash
export OPENROUTER_MODEL="google/gemini-2.5-flash"
export TORCH_DEVICE="cuda:1"
export FLASK_SECRET_KEY="replace-this-for-shared-deployments"
export PORT=5002
export USER_ID=0
export CONDITION_ID=0
export TASK_ID=0
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

The app loads Stable Diffusion XL from Hugging Face through `diffusers`, so the first run may download model weights.

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
