# Human-in-the-Loop Prompt Optimization

This repository contains the cleaned release of the image user-study app for human-in-the-loop prompt optimization.

## Overview

The app supports iterative human preference feedback for text-to-image generation:

1. A user enters an initial prompt.
2. The app generates image variants.
3. The user selects preferred images.
4. The method updates prompts using preference-guided refinement and evolutionary variation.
5. The loop continues until the user is satisfied.

## App

The runnable app is in:

```text
image_user_study_ours/
```

See [image_user_study_ours/README.md](image_user_study_ours/README.md) for setup, configuration, and run instructions.

## Requirements

This project requires Python and a machine capable of running Stable Diffusion XL through PyTorch. A CUDA-capable GPU is recommended for practical image-generation speed.
