import requests
import base64
from utils import api_key, model
import numpy as np
from sentence_transformers import SentenceTransformer
from utils import device
from scipy.special import softmax
from scipy.stats import entropy
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances

def crossover(prompt1, prompt2):
    content = [{
        "type": "text",
        "text": (
            f'''You are an AI assistant simulating crossover for evolutionary prompt optimization in a text-to-image generation task.

            Each parent prompt describes the **same object or scene**, but in a **different visual style** (e.g., art medium, color palette, mood, lighting, texture, rendering technique, etc.).

            The goal is to generate **three child prompts** that:
            1. **Preserve the object or scene** described in both parent prompts
            2. **Blend, recombine, or hybridize the stylistic elements** of the parents
            3. Explore **diverse combinations** of style modifiers (without assuming a fixed target style)
            4. Avoid exact repetition of full parent prompts
            5. Can serve as exploratory candidates for downstream evaluation (e.g., human preference, image similarity, aesthetic score)
            6. Each child prompt is under 60 words.
            
            Be creative and descriptive. Focus on **visual and stylistic traits** such as mood, lighting, rendering, detail level, color scheme, texture, medium, or technique. You may keep or alter the style names (e.g., "steampunk", "cyberpunk", "oil painting", etc.), or omit them entirely in favor of descriptive traits.

            ---

            Example Input:

            Parent A: "a futuristic city skyline at dusk, rendered in low-poly 3D style with pastel colors"  
            Parent B: "a futuristic city skyline at dusk, digital painting with painterly brushstrokes and glowing lights"

            Output:
            Child 1: "a futuristic city skyline at dusk, a fusion of low-poly geometry and painterly textures, glowing softly in pastel tones"  
            Child 2: "a futuristic city skyline at dusk, rendered in semi-abstract style with blocky shapes and expressive lighting"  
            Child 3: "a futuristic city skyline at dusk, with glowing brushstrokes, soft gradients, and minimalist polygonal forms"

            ---

            Now, process the following two parents and output 1 new, diverse child prompts:
            Prompt 1: {prompt1}
            Prompt 2: {prompt2}

            Only return the child prompt.
        '''
        )
    }]
    
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": content
            }
        ]
    }
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "http://localhost",
        "X-Title": "sentence-crossover"
    }
    
    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
    response.raise_for_status()
    
    lm_output = response.json()['choices'][0]['message']['content'].strip()
    return lm_output

def mutate_group_with_intensity(prompt, child_prompt_num, mutate_intensity):
    meta_prompt = f"""
        You are an AI assistant simulating **mutation** in an evolutionary algorithm for optimizing text-to-image prompts.

        ---

        ## Task
        Given **one input prompt** describing a specific object or scene, generate **{child_prompt_num} mutated versions** by changing its **stylistic elements** while keeping the **core subject** exactly the same.

        ---

        ## Mutation Rules

        ### 1. Preserve the Subject
        - The main object or scene described in the original prompt must remain unchanged.
        - Do **not** add, remove, or replace the main subject or setting.

        ### 2. Modify Style-Related Elements
        Make changes only to style-oriented descriptors, such as:
        - **Color palette** (e.g., warm earthy tones → neon pastels)
        - **Lighting** (e.g., soft morning light → dramatic chiaroscuro)
        - **Rendering technique** (e.g., watercolor → photorealism)
        - **Medium** (e.g., pencil sketch → digital painting)
        - **Mood/atmosphere** (e.g., serene → chaotic)
        - **Texture** (e.g., smooth glassy → rough, grainy)
        - **Level of abstraction** (e.g., hyperrealistic → minimalistic)
        - **Other visual descriptors** relevant to style

        ### 3. Control the Degree of Change with `intensity`
        The `intensity` parameter determines **how far** the mutation deviates from the original style:
        - **intensity ≈ 0** → Minimal stylistic change  
        - Alter **only one** style element slightly.  
        - Keep most wording and details identical.  
        - Small, subtle variations.  

        - **intensity ≈ 1** → Maximum stylistic change (**totally random variation**)  
        - Change **as many style elements as possible** while keeping the subject recognizable.  
        - Explore **completely different artistic styles, lighting, moods, mediums, and composition choices**.  
        - Use **different vocabulary, sentence structure, and tone** for each output.  
        - Every mutated prompt should feel **radically different** in style from both the original and from each other.

        ---

        ## Examples

        **Example Input Prompt:**  
        "A medieval castle on a hill at sunrise, painted in watercolor with soft pastel colors."

        **intensity = 0** (minimal change examples):  
        1. "A medieval castle on a hill at sunrise, painted in watercolor with slightly warmer golden tones."  
        2. "A medieval castle on a hill at sunrise, painted in watercolor with a cooler, misty color palette."

        **intensity = 1** (totally random style examples):  
        1. "In a futuristic neon cityscape, a towering medieval castle rises above the skyline, glowing in electric blues and magentas, rendered in glitch-art style."  
        2. "A dreamlike medieval castle perched high on a hill, sculpted entirely from molten glass, catching the light in fractal rainbows, in ultra-detailed 8K realism."  
        3. "An abstract cubist interpretation of a medieval castle on a hill, broken into geometric shards of crimson, gold, and midnight blue."  
        4. "A whimsical claymation scene of a medieval castle, its turrets swaying slightly under a pink cotton-candy sky."  
        5. "A dark, cinematic shot of a medieval castle during a raging storm, lit only by flashes of lightning, rendered in gritty black-and-white film grain."

        ---

        ## Output Format
        - Generate exactly **{child_prompt_num}** mutated prompts.
        - Each prompt should be on its own line.
        - Do **not** include numbering, bullets, or explanations.
        - Under 60 words.

        ---

        Current mutation intensity: {mutate_intensity}

        **Input Prompt to Mutate:**  
        {prompt}
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

    return prompt_variants

def cosine_similarity_vector(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def compute_similarity_to_parents(child_embs, parent_embs, strategy="max"):
    """Compute similarity of each child embedding to multiple parents."""
    sims = []
    for c in child_embs:
        sim_scores = [cosine_similarity_vector(c, p) for p in parent_embs]
        if strategy == "max":
            sims.append(np.max(sim_scores))
        elif strategy == "avg":
            sims.append(np.mean(sim_scores))
        else:
            raise ValueError("strategy must be 'max' or 'avg'")
    return np.array(sims)

def compute_diversity_scores(child_embs):
    n = len(child_embs)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                dist_matrix[i, j] = np.nan
            else:
                dist_matrix[i, j] = 1 - cosine_similarity_vector(child_embs[i], child_embs[j])
    return np.nanmean(dist_matrix, axis=1)

def min_max_normalize(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-8)

def pareto_frontier(S, D):
    points = np.stack((S, D), axis=1)
    is_dominated = np.zeros(len(points), dtype=bool)

    for i, p in enumerate(points):
        for j, q in enumerate(points):
            if i != j:
                # q dominates p if q is >= in all and > in at least one
                if (q >= p).all() and (q > p).any():
                    is_dominated[i] = True
                    break

    frontier_indices = np.where(~is_dominated)[0]
    return frontier_indices

def select_k_from_pareto(S, D, child_prompts, K):
    # Normalize scores
    S_norm = min_max_normalize(S)
    D_norm = min_max_normalize(D)

    # Step 1: Find Pareto frontier
    frontier_indices = pareto_frontier(S_norm, D_norm)

    if len(frontier_indices) >= K:
        # Step 2: Randomly choose K from the frontier
        selected = np.random.choice(frontier_indices, size=K, replace=False)
    else:
        # Step 3: Add all frontier points
        selected = list(frontier_indices)

        # Find all non-frontier indices
        all_indices = set(range(len(S)))
        non_frontier_indices = list(all_indices - set(frontier_indices))

        # Step 4: Compute distance from each non-frontier point to the frontier
        frontier_points = np.stack([S_norm[frontier_indices], D_norm[frontier_indices]], axis=1).T
        non_frontier_points = np.stack([S_norm[non_frontier_indices], D_norm[non_frontier_indices]], axis=1).T

        # Compute distance from each non-frontier point to the closest frontier point
        distances = []
        for nf_point in non_frontier_points.T:
            dists = np.linalg.norm(frontier_points.T - nf_point, axis=1)
            distances.append(np.min(dists))
        
        # Step 5: Select closest points to the frontier
        closest_indices = np.argsort(distances)[:K - len(selected)]
        selected += [non_frontier_indices[i] for i in closest_indices]

    # Return the selected prompts and optionally their scores
    selected_prompts = [child_prompts[i] for i in selected]
    selected_scores = [(S[i], D[i]) for i in selected]
    
    return selected_prompts, selected_scores

def select_children(parent_prompts, child_prompts, similarity_strategy="avg", num_variants=9):
    if isinstance(parent_prompts, str):
        parent_prompts = [parent_prompts]

    sentence_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

    parent_embs = sentence_model.encode(parent_prompts, convert_to_numpy=True)
    child_embs = sentence_model.encode(child_prompts, convert_to_numpy=True)

    S = compute_similarity_to_parents(child_embs, parent_embs, strategy=similarity_strategy)
    D = compute_diversity_scores(child_embs)

    K = num_variants - len(parent_prompts)
    selected_prompts, selected_scores = select_k_from_pareto(S, D, child_prompts, K)
    return selected_prompts, selected_scores