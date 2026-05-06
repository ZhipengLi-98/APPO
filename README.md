# Human-in-the-Loop Prompt Optimization

This sharing branch is centered on the final image user-study app for the proposed method.

## Final App

Use:

```text
image_user_study_ours/
```

The app supports iterative human preference feedback for text-to-image prompt optimization:

1. A user enters an initial prompt.
2. The app generates image variants.
3. The user selects preferred images.
4. The method updates prompts using preference-guided refinement and evolutionary variation.
5. The loop continues until the user is satisfied.

See [image_user_study_ours/README.md](image_user_study_ours/README.md) for setup and run instructions.

## Environment

All project operations should run in the `py39` conda environment:

```bash
conda activate py39
```

## Note

Other folders contain experiment scripts, baselines, analysis code, and archived artifacts from development. They are not needed to run the final shared app.
