from openai import OpenAI
keys = os.environ["OPENAI_API_KEY"]
# Replace with your actual API key string or env var
client = OpenAI(api_key=keys)
#load json file 
import json
import os
path='final_dataset.json'
#load json file 
with open(path, 'r') as f:
    data = json.load(f)
train_folder_path=os.path.join(os.getcwd(),'endo_train')
test_folder_path=os.path.join(os.getcwd(),'endo_test')

example = data['test'][:10]
example_img_path=[os.path.join(test_folder_path, ex['img_path']) for ex in example]
gt_cvs = [ex['CVS'] for ex in example]
print(gt_cvs)


import random
from itertools import product
from collections import defaultdict
from typing import Dict, List, Tuple, Any

Combo = Tuple[int, int, int]

def all_combos_3() -> List[Combo]:
    """All 2^3 = 8 multi-label combinations for 3 classes."""
    return list(product([0, 1], repeat=3))

def bucket_by_combo(train_data: List[Dict[str, Any]]) -> Dict[Combo, List[Dict[str, Any]]]:
    """
    Group frame dicts by their CVS vector.
    Returns: {combo: [frame_dicts]}
    """
    buckets: Dict[Combo, List[Dict[str, Any]]] = defaultdict(list)
    for frame in train_data:
        vec = frame["CVS"]
        if not isinstance(vec, (list, tuple)) or len(vec) != 3 or any(v not in (0, 1) for v in vec):
            raise ValueError(f"Invalid CVS for id={frame['id']}: {vec}")
        buckets[tuple(vec)].append(frame)
    return buckets

def select_per_combo_from_train(
    train_data: List[Dict[str, Any]],
    k: int = 1,
    seed: int = 42,
) -> Tuple[Dict[Combo, List[Dict[str, Any]]], List[Combo]]:
    """
    From train_data, randomly select up to k frames for each CVS combo.
    Returns:
      - samples_dict: {combo: [frame_dicts]}
      - missing_combos: list of combos with no examples
    """
    rng = random.Random(seed)
    buckets = bucket_by_combo(train_data)

    out: Dict[Combo, List[Dict[str, Any]]] = {}
    missing: List[Combo] = []

    for combo in all_combos_3():
        frames = buckets.get(combo, [])
        if not frames:
            out[combo] = []
            missing.append(combo)
            continue
        if len(frames) <= k:
            frames_copy = frames[:]
            rng.shuffle(frames_copy)
            out[combo] = frames_copy
        else:
            out[combo] = rng.sample(frames, k)

    return out, missing


# =========================
# Example usage:
# =========================

samples, missing = select_per_combo_from_train(data['train'], k=1, seed=42)



# Build few_shot_examples from samples
few_shot_examples = []

for combo, frames in samples.items():
    for f in frames:  # usually only 1 because k=1
        few_shot_examples.append(
            (os.path.join(train_folder_path,f["img_path"]), str(f["CVS"]))  # path + CVS label
        )

print("\nFew-shot examples:")
for ex in few_shot_examples:
    print(ex)
