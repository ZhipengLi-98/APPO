from flask import Flask, request, render_template, redirect, url_for, session
import threading, os, torch
from gradient import initial_mutate, get_next_prompts
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
from utils import device, IMAGE_HEIGHT, IMAGE_WIDTH

# pipe = StableDiffusionXLPipeline.from_pretrained(
#         "stabilityai/stable-diffusion-xl-base-1.0",
#         torch_dtype=torch.float16,
#         variant="fp16",
#         use_safetensors=True
#     )
# pipe.to(device)
# pipe.set_progress_bar_config(disable=True)

model_id = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = StableDiffusionXLPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
# pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device)

# Helper for generating images on GPU
def _generate_on_device(prompts, batch=False):
    global pipe
    with torch.inference_mode():
        if not batch:
            return [pipe(prompts).images[0]]
        results = pipe(prompts, height=IMAGE_HEIGHT, width=IMAGE_WIDTH,
                       num_inference_steps=50, guidance_scale=7.5)
        return results.images

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key")

OUTPUT_DIR = "static/generated"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Track generated images per user and iteration
generated_images = {}  # {user_id: {iteration_id: [(filename, prompt), ...]}}

# Auto-incrementing user/iteration counters (in production, better to store per session)
user_id = int(os.environ.get("USER_ID", "0"))
condition_id = int(os.environ.get("CONDITION_ID", "0"))
task_id = int(os.environ.get("TASK_ID", "0"))
iteration_id = 0

initial_prompt = ""

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        base_prompt = request.form["prompt"]
        session["base_prompt"] = base_prompt
        session["initial_prompt"] = base_prompt  # save initial prompt in session
        generated_images = {}
        print(base_prompt)
        return redirect(url_for("loading"))
    return render_template("index.html")
    
def save_prompts(iteration_id, prompts):
    user_iter_dir = os.path.join(OUTPUT_DIR, f"user_{user_id}_condition_{condition_id}_task_{task_id}", f"iteration_{iteration_id}")
    os.makedirs(user_iter_dir, exist_ok=True)

    txt_path = os.path.join(user_iter_dir, "prompts.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for prompt in prompts:
            f.write(f"{prompt}\n")  # tab-separated filename and prompt

def generate_images_for_iteration_initial(base_prompt):
    global user_id, condition_id, task_id, iteration_id, generated_images

    prompts = initial_mutate(base_prompt, 9)
    prompts = prompts[:9]
    save_prompts(iteration_id, prompts)
    import time
    start_time = time.time()
    print(f"Generating {len(prompts)} images in a batch...")

    images = _generate_on_device(prompts, batch=True)
    print(f"Generation completed (elapsed {time.time() - start_time:.2f}s)")

    user_folder = os.path.join(OUTPUT_DIR, f"user_{user_id}_condition_{condition_id}_task_{task_id}", f"iteration_{iteration_id}")
    os.makedirs(user_folder, exist_ok=True)

    image_files = []
    for img_id, (prompt, image) in enumerate(zip(prompts, images), start=1):
        filename = f"img{img_id}.png"
        filepath = os.path.join(user_folder, filename)
        image.save(filepath)
        image_files.append((os.path.relpath(filepath, OUTPUT_DIR), prompt))

    temp = f"user_{user_id}_condition_{condition_id}_task_{task_id}"

    if temp not in generated_images:
        generated_images[temp] = {}
    generated_images[temp][iteration_id] = image_files

@app.route("/loading")
def loading():
    base_prompt = session.get("base_prompt")
    if not base_prompt:
        return redirect(url_for("index"))

    # Kick off background generation in a thread
    threading.Thread(target=generate_images_for_iteration_initial, args=(base_prompt,)).start()
    return render_template("loading.html")

@app.route("/status")
def status():
    global user_id, condition_id, task_id, iteration_id
    temp = f"user_{user_id}_condition_{condition_id}_task_{task_id}"
    if temp in generated_images and iteration_id in generated_images[temp]:
        return {"ready": True}
    return {"ready": False}

@app.route("/prompt_status")
def prompt_status():
    global user_id, condition_id, task_id, iteration_id
    temp = f"user_{user_id}_condition_{condition_id}_task_{task_id}"
    key = (temp, iteration_id)
    if key in prompt_cache:
        return {"ready": True}
    return {"ready": False}

def save_selected_images(iteration_id, selected_files, selected_prompts):
    """Save the names of selected images and their prompts to a text file in the iteration folder."""
    import os
    global user_id, condition_id, task_id

    user_iter_dir = os.path.join(OUTPUT_DIR, f"user_{user_id}_condition_{condition_id}_task_{task_id}", f"iteration_{iteration_id}")
    os.makedirs(user_iter_dir, exist_ok=True)

    txt_path = os.path.join(user_iter_dir, "selected_images.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for filename, prompt in zip(selected_files, selected_prompts):
            f.write(f"{filename}\t{prompt}\n")  # tab-separated filename and prompt

    print(f"Saved selected images and prompts to {txt_path}")

def generate_prompts_for_iteration(selected_prompts, all_prompts, initial_prompt):
    """Step 1: generate new prompts."""
    return get_next_prompts(selected_prompts, all_prompts, initial_prompt, iteration_id)

def generate_images_for_iteration(new_prompts):
    """Step 2: generate images for given prompts."""
    global user_id, condition_id, task_id, iteration_id, generated_images

    import time
    start_time = time.time()
    print(f"Generating {len(new_prompts)} images in a batch...")
    new_images = _generate_on_device(new_prompts, batch=True)
    print(f"Generation completed (elapsed {time.time() - start_time:.2f}s)")

    # Save images
    user_dir = os.path.join(OUTPUT_DIR, f"user_{user_id}_condition_{condition_id}_task_{task_id}", f"iteration_{iteration_id}")
    os.makedirs(user_dir, exist_ok=True)

    image_files = []
    for img_id, (prompt, image) in enumerate(zip(new_prompts, new_images), start=1):
        filename = f"img{img_id}.png"
        filepath = os.path.join(user_dir, filename)
        image.save(filepath)
        image_files.append((os.path.join(f"user_{user_id}_condition_{condition_id}_task_{task_id}/iteration_{iteration_id}", filename), prompt))

    temp = f"user_{user_id}_condition_{condition_id}_task_{task_id}"
    if temp not in generated_images:
        generated_images[temp] = {}
    generated_images[temp][iteration_id] = image_files

# At the top of app.py
prompt_cache = {}  # stores prompts per user/iteration

def run_prompt_phase(selected_prompts, all_prompts, initial_prompt, iteration_id):
    new_prompts = generate_prompts_for_iteration(selected_prompts, all_prompts, initial_prompt)
    new_prompts = new_prompts[:9]
    save_prompts(iteration_id, new_prompts)
    temp = f"user_{user_id}_condition_{condition_id}_task_{task_id}"
    prompt_cache[(temp, iteration_id)] = new_prompts

@app.route("/gallery", methods=["GET", "POST"])
def gallery():
    global user_id, iteration_id, generated_images

    initial_prompt = session.get("initial_prompt", "")  # get initial prompt from session

    if request.method == "POST":
        selected_files = request.form.getlist("selected")
        temp = f"user_{user_id}_condition_{condition_id}_task_{task_id}"
        images = generated_images.get(temp, {}).get(iteration_id, [])
        selected_prompts = [p for f, p in images if f in selected_files]
        save_selected_images(iteration_id, selected_files, selected_prompts)
        all_prompts = [prompt for _, prompt in images]

        satisfied = request.form.get("satisfied")
        if satisfied:
            user_iter_dir = os.path.join(OUTPUT_DIR, f"user_{user_id}_condition_{condition_id}_task_{task_id}", f"iteration_{iteration_id}")
            txt_path = os.path.join(user_iter_dir, "satisfied.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("satisfied\n")
            # You could also save this in a log file or database instead of just printing
            return redirect(url_for("thank_you"))

        iteration_id += 1

        # start background thread for prompts
        threading.Thread(
            target=run_prompt_phase,
            args=(selected_prompts, all_prompts, initial_prompt, iteration_id)
        ).start()

        return redirect(url_for("prompt_loading"))

    temp = f"user_{user_id}_condition_{condition_id}_task_{task_id}"
    images = generated_images.get(temp, {}).get(iteration_id, [])
    return render_template("gallery.html", images=images, initial_prompt=initial_prompt, iteration=iteration_id+1)

@app.route("/thank_you")
def thank_you():
    return render_template("thank_you.html")

@app.route("/image_loading")
def image_loading():
    global user_id, iteration_id
    temp = f"user_{user_id}_condition_{condition_id}_task_{task_id}"
    key = (temp, iteration_id)
    if key in prompt_cache:
        new_prompts = prompt_cache.pop(key)  # take and remove
        threading.Thread(
            target=generate_images_for_iteration,
            args=(new_prompts,)
        ).start()
    return render_template("image_loading.html")

@app.route("/prompt_loading")
def prompt_loading():
    return render_template("prompt_loading.html")

@app.route("/gallery_loading")
def gallery_loading():
    return render_template("gallery_loading.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5002"))
    app.run(host="0.0.0.0", port=port)
