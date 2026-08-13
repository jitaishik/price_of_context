import argparse
import copy
import json
import random
import sys
import traceback
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.llm_client import call_vllm_text

parser = argparse.ArgumentParser()
parser.add_argument("--vllm-host",   default="localhost")
parser.add_argument("--vllm-port",   default=8000, type=int)
parser.add_argument("--model",       default="path_to_model")
parser.add_argument("--input",       default="eeyore-data.parquet")
parser.add_argument("--valid-idx",   default="valid_indices.json")
parser.add_argument("--output",      default="eeyore-dp-simple.parquet")
parser.add_argument("--keep-prob",   default=0.7, type=float,
                    help="Probability of keeping each attribute unchanged (default: 0.7)")
parser.add_argument("--seed",        default=42, type=int)
args = parser.parse_args()

VLLM_BASE = f"http://{args.vllm_host}:{args.vllm_port}/v1"
MODEL     = args.model
P_KEEP    = args.keep_prob
P_ADD     = 1.0 - P_KEEP
random.seed(args.seed)

PERTURB_ATTRS = ["age", "gender", "occupation", "marital status"]

FIXED_FIELDS = {
    "resistance toward the support",
    "counseling history",
}

SYMPTOM_FIELDS = {
    "symptom severity",
    "cognition distortion exhibition",
    "depression severity",
    "suicidal ideation severity",
    "homicidal ideation severity",
}

print(f"Loading {args.input} ...")
df = pd.read_parquet(args.input)

with open(args.valid_idx) as f:
    valid_indices = set(json.load(f))

print(f"Total rows: {len(df):,}, Valid indices: {len(valid_indices)}")
print(f"Keep prob: {P_KEEP}, Add prob: {P_ADD}")

print("Building attribute pool ...")

pool: dict = {k: [] for k in PERTURB_ATTRS}
valid_profiles: dict = {}

for idx in valid_indices:
    try:
        p = json.loads(df.iloc[idx]["profile"])
        valid_profiles[idx] = p
        for attr in PERTURB_ATTRS:
            if attr in p and p[attr]:
                pool[attr].append(p[attr])
    except Exception as e:
        print(f"  Warning: parse error at idx={idx}: {e}")

print("Pool sizes:", {k: len(v) for k, v in pool.items()})


def perturb_attribute(attr: str, original, present: bool):
    if not present:
        if random.random() < P_ADD and pool[attr]:
            return random.choice(pool[attr]), True
        return None, False

    if random.random() < P_KEEP:
        return original, False

    candidates = [v for v in pool[attr] if v != original]
    if not candidates:
        candidates = pool[attr]
    new_val = random.choice(candidates) if candidates else original
    return new_val, (new_val != original)

def call_vllm(messages: list, max_tokens: int = 512, temperature: float = 0.7,
              retries: int = 3) -> str:
    return call_vllm_text(VLLM_BASE, MODEL, messages, max_tokens=max_tokens,
                           temperature=temperature, retries=retries)


def fmt_changes(orig: dict, new: dict) -> str:
    parts = []
    for attr in PERTURB_ATTRS:
        ov = orig.get(attr, "[not present]")
        nv = new.get(attr,  "[not present]")
        if ov != nv:
            parts.append(f"  - {attr}: {ov!r} -> {nv!r}")
    return "\n".join(parts) if parts else "  (none)"


def rewrite_situation(orig: dict, new: dict) -> str:
    symptom_lines = []
    for sf in SYMPTOM_FIELDS:
        if sf in new:
            val = new[sf]
            if isinstance(val, dict):
                val = json.dumps(val)
            symptom_lines.append(f"  {sf}: {val}")
    final_attrs = {k: v for k, v in new.items()
                   if k not in SYMPTOM_FIELDS | FIXED_FIELDS
                   and k != "situation of the client"}
    
    prompt = (
        "You are a careful text editor. Your job is to make minimal targeted edits to a client situation description when demographic attributes change.\n\n"
        "STRICT RULE: The output must have the same structure and information density as the original. "
        "Do NOT add any demographic descriptors (age, gender, occupation, marital status, etc.) that are not already present in the original text. "
        "The profile attributes are a background constraint for coherence only — they must NOT appear in the output text unless they were already in the input text.\n\n"
        f"ORIGINAL situation:\n{orig.get('situation of the client', 'N/A')}\n\n"
        f"ATTRIBUTE CHANGES:\n{fmt_changes(orig, new)}\n\n"
        f"FINAL PROFILE ATTRIBUTES (coherence check only — do NOT insert these into the text):\n"
        + "\n".join(f"  {k}: {v}" for k, v in final_attrs.items()) + "\n\n"
        "SYMPTOMS TO KEEP CONSISTENT (do NOT quote severity scores):\n"
        + "\n".join(symptom_lines) + "\n\n"
        "Editing rules:\n"
        "  - Only edit what is directly necessitated by the ATTRIBUTE CHANGES above\n"
        "  - Preserve all original phrasing that does not need to change\n"
        "  - Adapt specific circumstances to fit the new profile (e.g. if occupation changed,\n"
        "    replace work-related details with equivalent ones for the new occupation)\n"
        "  - Preserve all clinically and emotionally relevant content: the nature of the\n"
        "    psychological distress, relationship dynamics, interpersonal conflicts, and\n"
        "    life circumstances that are contextually important\n"
        "  - Do NOT add new sentences or details not present in the original — only adapt existing ones\n"
        "  - Do NOT expand on symptoms, cognitive distortions, or any detail not present in the original\n"
        "  - Keep the description in third-person and in roughly the same length and style\n"
        "  - Do NOT include numerical severity scores\n"
        "  - Return ONLY the edited text, no preamble."

    )
    return call_vllm([{"role": "user", "content": prompt}], max_tokens=350)

print(f"\nProcessing {len(valid_indices)} profiles ...\n")

df_out = df.copy()
n_modified  = 0
n_unchanged = 0
n_errors    = 0
sorted_valid = sorted(valid_indices)

for loop_i, idx in enumerate(sorted_valid):
    if idx not in valid_profiles:
        continue

    try:
        orig = valid_profiles[idx]
        new  = copy.deepcopy(orig)
        changed_attrs = []

        # Remove name if present
        if "name" in new:
            del new["name"]
            changed_attrs.append("name")

        # Perturb each attribute independently
        for attr in PERTURB_ATTRS:
            present = attr in orig
            new_val, changed = perturb_attribute(attr, orig.get(attr), present)
            if changed:
                changed_attrs.append(attr)
                if new_val is None:
                    new.pop(attr, None)
                else:
                    new[attr] = new_val

        if not changed_attrs:
            n_unchanged += 1
            continue

        # Rewrite situation if anything changed
        new["situation of the client"] = rewrite_situation(orig, new)

        # Safety check
        for sf in SYMPTOM_FIELDS | FIXED_FIELDS:
            if sf in orig and orig[sf] != new.get(sf):
                raise ValueError(f"BUG: field '{sf}' was altered at idx={idx}!")

        df_out.at[idx, "profile"] = json.dumps(new, indent=1)
        n_modified += 1

        if (loop_i + 1) % 10 == 0 or loop_i < 5:
            print(f"  [{loop_i+1:3d}/{len(sorted_valid)}] idx={idx:4d}  changed={changed_attrs}")

    except Exception as e:
        n_errors += 1
        print(f"  [{loop_i+1:3d}/{len(sorted_valid)}] idx={idx:4d}  ERROR: {e}")
        traceback.print_exc()

df_out.to_parquet(args.output, index=False)

print(f"""
{'='*45}
  Summary
{'='*45}
  Valid indices processed : {len(sorted_valid)}
  Profiles modified       : {n_modified}
  Profiles unchanged      : {n_unchanged}
  Errors                  : {n_errors}
  Output                  : {args.output}
{'='*45}
""")
