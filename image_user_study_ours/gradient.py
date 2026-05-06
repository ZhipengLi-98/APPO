import requests
import base64
from utils import api_key, model
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
from sentence_transformers import SentenceTransformer
from ea import crossover, mutate_group_with_intensity, select_children
from itertools import combinations
from tqdm import tqdm

sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

prompt_similarity_upper_bound = -1
prompt_similarity_lower_bound = 1

preferred_history = []
unpreferred_history = []

def normalize(x, old_min, old_max, new_min=0, new_max=1):
    if old_max == old_min:
        raise ValueError("Old max and min cannot be the same")
    return (x - old_min) / (old_max - old_min) * (new_max - new_min) + new_min

def initial_mutate(initial_prompt, variant_num=9):
    meta_prompt = f"""
        Write {variant_num} variants based on this prompt
        {initial_prompt}
        
        Each variant should:
        - Remain the objects and their composition in the old prompt
        - Explore different style directions, as much different as possible.
        - Be concise (under 60 words)
        - Include visually rich and specific details
        - Be diverse in composition and wording
        - Be suitable for models like Midjourney, DALL·E, or Stable Diffusion

        Output only the {variant_num} new prompts. Each on its own line. No explanation.
        """

    content = [{"type": "text", "text": meta_prompt}]
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}]
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "http://localhost",
        "X-Title": "global-variant-generation"
    }

    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
    response.raise_for_status()

    result_text = response.json()['choices'][0]['message']['content'].strip()
    prompt_variants = [line.strip("-•1234567890. ") for line in result_text.splitlines() if line.strip()]
    print(prompt_variants)
    return prompt_variants

def summarize_prompt_history(preferred_history, unpreferred_history):
    summarization_prompt = f"""
        You are an assistant that extracts **stylistic and aesthetic trends** from prompt examples.

        Below are previously preferred prompts:
        {preferred_history}

        Below are previously unpreferred prompts:
        {unpreferred_history}

        Your task:

        1. Identify the **distinct visual styles or aesthetics** (e.g., cyberpunk, golden hour, anime, photorealism, minimalism, surrealism, vintage, etc.) that appear in the preferred prompts. Be specific and name the styles.
        2. Identify which **specific styles or characteristics** appear frequently in the unpreferred prompts.
        3. Clearly conclude which styles are **consistently preferred** and which are **not preferred**, based on these examples.
        4. Summarize this as:
        - A short paragraph describing **which styles are most favored**, with examples of good visual traits (e.g., dramatic lighting, deep contrast, cinematic framing).
        - A short paragraph describing **which styles or traits are less favored or problematic**, with examples (e.g., flat lighting, generic scenery, overused tropes).
        5. Use clear, **style-labeled language** and avoid repeating full prompt texts.

        Format:
        Preferred style summary:
        ...

        Unpreferred style summary:
        ...

        Style preference conclusion:
        ...
    """

    content = [
        {
            "type": "text",
            "text": summarization_prompt
        }
    ]

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}]
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "http://localhost",
        "X-Title": "prompt-history-summary"
    }

    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
    response.raise_for_status()

    summary = response.json()["choices"][0]["message"]["content"]
    return summary

def self_reflect_prompt_consistency(initial_prompt, revised_prompt):
    reflection_prompt = f"""
        You are a helpful assistant for prompt evaluation and refinement.

        Your task is to:
        1. Reflect on whether the following revised prompt retains all important objects, relationships, and visual elements from the initial prompt.
        2. If anything important is missing, revise the prompt to add those missing elements while preserving clarity, conciseness (under 60 words), and visual richness.
        3. If nothing is missing, return the original revised prompt as-is.

        Initial prompt:
        "{initial_prompt}"

        Revised prompt:
        "{revised_prompt}"

        Only respond with the final prompt (either unchanged or improved to restore missing content)
    """

    content = [{"type": "text", "text": reflection_prompt}]
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}]
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "http://localhost",
        "X-Title": "self-reflection-check"
    }

    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
    response.raise_for_status()

    reflection = response.json()['choices'][0]['message']['content'].strip()
    return reflection

def generate_global_gradient_without_images(preferred_prompts, unpreferred_prompts, preferred_history=[], unpreferred_history=[]):

    # Construct the meta prompt
    meta_prompt = f"""
        You are assisting in refining image generation prompts based on user preferences.

        Users have shown a clear preference for the following prompts:
        {preferred_prompts}

        In contrast, the following prompts were not preferred:
        {unpreferred_prompts}
        """

    if preferred_history and unpreferred_history:
        summary = summarize_prompt_history(preferred_history, unpreferred_history)
        meta_prompt += f"""
            Summary of feedback from earlier iterations:
            {summary}
        """
        # print(summary)

    meta_prompt += """
        Please analyze the visual differences between the preferred and unpreferred image prompts, focusing especially on the **stylistic features and fine-grained visual aesthetics** that each prompt produces.

        Identify what **specific styles** (e.g., cinematic, minimalist, painterly, photorealistic, surreal, vintage, anime, etc.) or **visual characteristics** (e.g., lighting, texture, composition, color grading, camera angle, subject positioning, background complexity) are preferred.

        Compare these features against those in the unpreferred prompts, and describe what **key visual elements or stylistic patterns** are lacking or less desirable.

        Provide a clear, **style-focused** piece of feedback that reveals how future prompts can better align with the preferred visual outcomes.
        
        Respond only with the feedback, as it will be used as a global gradient signal to enhance prompt quality.
        """
    
    # if style_novelty_check(preferred_prompts, preferred_history):
    #     meta_prompt += "\nDetected stylistic convergence. Suggest more novel and diverse stylistic directions for exploration."

    content = [
        {
            "type": "text",
            "text": meta_prompt
        }
    ]

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}]
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "http://localhost",
        "X-Title": "prompt-gradient-with-image"
    }

    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
    response.raise_for_status()

    gradient = response.json()['choices'][0]['message']['content'].strip()

    return {
        "global_gradient": gradient,
        "preferred_prompts": preferred_prompts,
        "unpreferred_prompts": unpreferred_prompts
    }

def incorporate_global_gradient_without_images(gradients, preferred_prompts, unpreferred_prompts, initial_prompt, num_variants=9):
    updated_prompts = {}

    # Build the meta prompt
    meta_prompt = f"""
        You are an expert prompt engineer for text-to-image generation. Your task is to rewrite and improve the original prompt to create more preferred image outputs.

        The following prompts were not preferred by users:
        {unpreferred_prompts}

        Summarized feedback on why these prompts may be suboptimal:
        {gradients["global_gradient"]}

        Using this feedback, generate {num_variants - len(preferred_prompts)} improved prompt variants that:
        - Retain all key objects and their arrangement from the original prompt: {initial_prompt}
        - Draw inspiration from the original unpreferred prompts
        - Address the common issues highlighted in the feedback
        - Are concise (under 60 words)
        - Include vivid, specific, and visually rich details
        - Use feedback constructively, but do not overfit — allow **creative detours**
        - Are suitable for models like Midjourney, DALL·E, or Stable Diffusion

        Output exactly {num_variants - len(preferred_prompts)} revised prompts, each on its own line, with no explanation.
    """

    content = [{"type": "text", "text": meta_prompt}]
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}]
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "http://localhost",
        "X-Title": "global-variant-generation"
    }

    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
    response.raise_for_status()

    result_text = response.json()['choices'][0]['message']['content'].strip()
    prompt_variants = [line.strip("-•1234567890. ") for line in result_text.splitlines() if line.strip()]
    # print(prompt_variants)
    res = {}
    original_prompts = unpreferred_prompts
    # print(len(original_prompts), len(prompt_variants))
    for i in range(len(prompt_variants)):
        if i >= len(original_prompts):
            continue
        prompt_variants[i] = self_reflect_prompt_consistency(initial_prompt, prompt_variants[i])
        res[original_prompts[i]] = {"new_prompt": prompt_variants[i], "original_prompt": None, "gradient": gradients["global_gradient"]}
    
    # --- Handle preferred prompts: Paraphrase only ---
    for preferred_prompt in preferred_prompts:
        res[preferred_prompt] = {
            "original_prompt": preferred_prompt,
            "gradient": gradients["global_gradient"],
            "new_prompt": preferred_prompt
        }

    return res

def get_next_prompts(selected_prompts, all_prompts, initial_prompt, cur_iter):
    global prompt_similarity_upper_bound, prompt_similarity_lower_bound, sentence_modelpreferred_history, unpreferred_history

    embeddings = sentence_model.encode(all_prompts, convert_to_numpy=True, normalize_embeddings=True)

    # Average pairwise cosine similarity
    pairs = list(combinations(embeddings, 2))
    avg_similarity = np.mean([np.dot(a, b) for a, b in pairs])
    if avg_similarity < prompt_similarity_lower_bound:
        prompt_similarity_lower_bound = avg_similarity
    if avg_similarity > prompt_similarity_upper_bound:
        prompt_similarity_upper_bound = avg_similarity

    # Default values
    gradient_num = 0
    ea_num = 0

    # Mapping for deterministic cases
    fixed_mapping = {
        1: (4, 4),
        3: (3, 3),
        5: (2, 2),
        7: (1, 1),
    }

    # Apply mapping if top_images has a fixed pair
    if len(selected_prompts) in fixed_mapping:
        gradient_num, ea_num = fixed_mapping[len(selected_prompts)]

    # Handle cases with randomness
    elif len(selected_prompts) == 2:
        if random.random() > 0.5:
            gradient_num, ea_num = 3, 4
        else:
            gradient_num, ea_num = 4, 3
    elif len(selected_prompts) == 4:
        if random.random() > 0.5:
            gradient_num, ea_num = 2, 3
        else:
            gradient_num, ea_num = 3, 2
    elif len(selected_prompts) == 6:
        if random.random() > 0.5:
            gradient_num, ea_num = 1, 2
        else:
            gradient_num, ea_num = 2, 1
    elif len(selected_prompts) == 8:
        if random.random() > 0.5:
            gradient_num, ea_num = 0, 1
        else:
            gradient_num, ea_num = 1, 0

    unpreferred_prompts = [i for i in all_prompts if i not in selected_prompts]

    print("Generate Gradients")
    initial_gradients = generate_global_gradient_without_images(selected_prompts, unpreferred_prompts, preferred_history, unpreferred_history)
    print("Gradients Generation Done")

    print("Incorporate Gradients")
    updated_prompts = incorporate_global_gradient_without_images(initial_gradients, selected_prompts, unpreferred_prompts, initial_prompt=initial_prompt, num_variants=gradient_num + len(selected_prompts))
    print("Gradients Incorporation Done")

    preferred_history.extend(selected_prompts)
    unpreferred_history.extend(unpreferred_prompts)
    
    after_gradient_prompts = [i["new_prompt"] for i in updated_prompts.values()]
    mutate_intensity = 0
    if cur_iter > 1:
        # current prompts similarity
        after_gradient_prompts.extend(selected_prompts)
        embeddings = sentence_model.encode(after_gradient_prompts, convert_to_numpy=True, normalize_embeddings=True)

        # Average pairwise cosine similarity
        pairs = list(combinations(embeddings, 2))
        avg_similarity = np.mean([np.dot(a, b) for a, b in pairs])

        mutate_intensity = normalize(avg_similarity, prompt_similarity_lower_bound, prompt_similarity_upper_bound)
        print(f"Current similarity before normalization: {avg_similarity}, after normalization: {mutate_intensity}, upper bound: {prompt_similarity_upper_bound}, lower bound: {prompt_similarity_lower_bound}")

    crossover_prompt = []
    population_prompts = selected_prompts
    population_size = len(population_prompts)

    print("Srart Crossover")
    if len(population_prompts) > 1:
        for _ in range(5):
            i, j = random.sample(range(population_size), 2)

            p1, p2 = population_prompts[i], population_prompts[j]
            child_prompt = crossover(p1, p2)
            child_prompt = self_reflect_prompt_consistency(initial_prompt, child_prompt)
            crossover_prompt.append(child_prompt)
        print("Crossover Done")

        print("Start Muatation")
        mutated_prompts = []
        for prompt in crossover_prompt:
            mutated = mutate_group_with_intensity(prompt, 2, mutate_intensity)
            for prompt in mutated:
                mutated = self_reflect_prompt_consistency(initial_prompt, prompt)
                # print(prompt)
                # print(mutated)
                mutated_prompts.append(mutated)
        print("Mutation Done")
    else:
        crossover_prompt = population_prompts
        print("Crossover Done")

        print("Start Muatation")
        mutated_prompts = []
        temp_mutate = mutate_group_with_intensity(crossover_prompt, 10, mutate_intensity)
        for prompt in temp_mutate:
            mutated = self_reflect_prompt_consistency(initial_prompt, prompt)
            # print(prompt)
            # print(mutated)
            mutated_prompts.append(mutated)
        print("Mutation Done")

    children_prompts, selected_scores = select_children(selected_prompts, mutated_prompts, num_variants=ea_num + len(selected_prompts))

    for key, val in updated_prompts.items():
        if val["original_prompt"] is None:
            children_prompts.append(val["new_prompt"])

    children_prompts.extend(selected_prompts)

    return children_prompts