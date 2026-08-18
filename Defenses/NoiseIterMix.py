import argparse
import copy
import json
import random
import sys
import traceback
from pathlib import Path
import numpy as np

from sentence_transformers import SentenceTransformer
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.llm_client import call_vllm_text, get_vllm_client_for_model, VLLM_BASE_URL

REPO_ROOT = Path(__file__).resolve().parents[1]
EEYORE_DATA_PATH = REPO_ROOT / "eeyore-data.parquet"
VALID_INDICES_PATH = REPO_ROOT / "valid_indices.json"

parser = argparse.ArgumentParser()
parser.add_argument("-model", "--model_name", type=str, default="llama",
                    help="Name of the model used for generation.")
parser.add_argument("-o", "--output",        default="eeyore-dp-mix-iter.parquet")
parser.add_argument("-p", "--noise-prob",     default=0.7, type=float,
                    help="Probability of keeping each attribute unchanged (default: 0.7)")
parser.add_argument("--seed",          default=42, type=int)
parser.add_argument("-th", "--sim-threshold", default=0.6, type=float,
                    help="Stop iterating once cosine similarity to original <= this (default: 0.6)")
parser.add_argument("--max-iters",     default=5, type=int,
                    help="Maximum divergence iterations on the situation text (default: 5)")
parser.add_argument("--st-model",      default="all-MiniLM-L6-v2",
                    help="Sentence-transformer model for similarity (default: all-MiniLM-L6-v2)")
parser.add_argument("--len-tol",       default=1.15, type=float,
                    help="Max allowed output length as a multiple of the original word count; "
                         "longer outputs are condensed back (default: 1.15)")
parser.add_argument("--no-references", action="store_true",
                    help="Disable attribute-matched reference situations in the divergence step")
args = parser.parse_args()

VLLM_BASE = VLLM_BASE_URL
MODEL, _  = get_vllm_client_for_model(args.model_name)
P_KEEP    = 1 - args.noise_prob
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

# Ordered age brackets — "similar" = same or adjacent bracket
AGE_BRACKETS = ["0~24", "25~44", "45~64", "65+"]

# Occupation clusters — "similar occupation" = same cluster
OCCUPATION_CLUSTERS = [
    ["Academic Research Associate", "PhD in immunology", "PhD student", "Medical student",
     "Graduate in a field related to their master's degree",
     "Recent college graduate looking for a job in economic research",
     "Film Student and Podcast Host","Student", "College student", "college student", "Intern"],
    ["Computer Science Engineer", "Computer stuff", "Electrical engineer",
     "Engineer in Mechanical Engineering", "Tech person", "Technical support specialist",
     "Customer service and technical expert"],
    ["Doctor", "Retired Nurse"],
    ["Special education teacher", "Assistant religious ed teacher and daycare worker"],
    ["Accounting. Low entry, bookkeeper.", "Bank employee", "Teller at a bank",
     "Employee in a company, potentially managerial or administrative", "Commission based job"],
    ["Retail", "Home Depot employee", "Kmart employee", "McDonald's employee",
     "Gas station worker", "Call center employee", "Hotel or Conference Center Worker",
     "Security worker", "hair braider", "Cleaning job"],
    ["Writer", "Drummer", "Music industry professional"],
    ["Military", "Peace Corps volunteer", "Non-profit library system employee",
     "Self employed", "Works from home", "Part-time job"],
    ["Unemployed"]
]
_OCC_TO_CLUSTER: dict = {}
for _ci, _cluster in enumerate(OCCUPATION_CLUSTERS):
    for _occ in _cluster:
        _OCC_TO_CLUSTER[_occ] = _ci
print(f"Loading sentence-transformer '{args.st_model}' ...")
model = SentenceTransformer(args.st_model)


def cosine_similarity(a: str, b: str) -> float:
    """Cosine similarity between two texts using the sentence-transformer."""
    emb = model.encode([a, b])
    va, vb = np.asarray(emb[0]), np.asarray(emb[1])
    va = va / (np.linalg.norm(va) + 1e-12)
    vb = vb / (np.linalg.norm(vb) + 1e-12)
    return float(va @ vb)
print(f"Loading {EEYORE_DATA_PATH} ...")
df = pd.read_parquet(EEYORE_DATA_PATH)

with open(VALID_INDICES_PATH) as f:
    valid_indices = set(json.load(f))

print(f"Total rows: {len(df):,}, Valid indices: {len(valid_indices)}")
print(f"Keep prob: {P_KEEP}, Add prob: {P_ADD}")
print(f"Sim threshold: {args.sim_threshold}, Max iters: {args.max_iters}")
print("Building attribute pool ...")

pool: dict = {k: [] for k in PERTURB_ATTRS}
valid_profiles: dict = {}

from collections import defaultdict
situation_index: dict = defaultdict(lambda: defaultdict(list))

for idx in valid_indices:
    try:
        p = json.loads(df.iloc[idx]["profile"])
        valid_profiles[idx] = p
        sit = p.get("situation of the client", "")
        for attr in PERTURB_ATTRS:
            if attr in p and p[attr]:
                pool[attr].append(p[attr])
                if sit:
                    situation_index[attr][p[attr]].append((idx, sit))
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


def retrieve_reference(new: dict, source_idx: int):
    if args.no_references:
        return None

    options = []  # (attr, value, [situations])

    for attr in PERTURB_ATTRS:
        val = new.get(attr)
        if val is None:
            continue

        if attr == "occupation":
            ci = _OCC_TO_CLUSTER.get(val, -1)
            sits = []
            if ci != -1:
                for occ in OCCUPATION_CLUSTERS[ci]:
                    sits += [s for (i, s) in situation_index["occupation"].get(occ, [])
                             if i != source_idx]
            else:
                sits = [s for (i, s) in situation_index["occupation"].get(val, [])
                        if i != source_idx]
        elif attr == "age":
            sits = []
            if val in AGE_BRACKETS:
                ai = AGE_BRACKETS.index(val)
                near = [val]
                if ai > 0:                    near.append(AGE_BRACKETS[ai - 1])
                if ai < len(AGE_BRACKETS) - 1: near.append(AGE_BRACKETS[ai + 1])
                for b in near:
                    sits += [s for (i, s) in situation_index["age"].get(b, [])
                             if i != source_idx]
            else:
                sits = [s for (i, s) in situation_index["age"].get(val, [])
                        if i != source_idx]
        else:  # gender, marital status — exact match
            sits = [s for (i, s) in situation_index[attr].get(val, [])
                    if i != source_idx]

        if sits:
            options.append((attr, val, sits))

    if not options:
        return None

    attr, val, sits = random.choice(options)
    return {"attr": attr, "value": val, "situation": random.choice(sits)}


def enforce_length(text: str, target_words: int) -> str:
    if target_words <= 0:
        return text
    if len(text.split()) <= target_words * args.len_tol:
        return text
    prompt = (
        f"Condense the following client situation to about {target_words} words "
        "(it is currently too long). Preserve the core issue, all clinically relevant "
        "content, and third-person voice. Do NOT add anything. Remove only redundancy and "
        "padding. Return ONLY the condensed text, no preamble.\n\n"
        f"TEXT:\n{text}"
    )
    return call_vllm([{"role": "user", "content": prompt}], max_tokens=300, temperature=0.3)


def adapt_situation(orig: dict, new: dict, target_words: int) -> str:
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
        f"ORIGINAL situation (~{target_words} words):\n{orig.get('situation of the client', 'N/A')}\n\n"
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
        f"  - Keep the description in third-person and at about {target_words} words (do NOT make it longer)\n"
        "  - Do NOT include numerical severity scores\n"
        "  - Return ONLY the edited text, no preamble."
    )
    return call_vllm([{"role": "user", "content": prompt}], max_tokens=350)


def diverge_situation(current: str, final_attrs: dict, symptom_lines: list[str],
                      target_words: int, reference: dict | None) -> str:
    reference_block = ""
    if reference is not None:
        reference_block = (
            f"\nREFERENCE situation (from another client who shares "
            f"{reference['attr']} = {reference['value']!r}). You may either borrow ONE small, "
            f"{reference['attr']}-appropriate circumstantial detail from it, OR adopt its "
            f"writing style/phrasing — whichever keeps this client coherent. Do NOT copy it "
            f"wholesale and do NOT change this client's core issue:\n"
            f"  {reference['situation']}\n"
        )

    prompt = (
        "You are lightly revising a client situation description for a mental health research dataset.\n\n"
        "Reword the text below and, where natural, make SMALL substitutions of peripheral details "
        "so the text reads differently from an earlier version. The goal is to reduce textual overlap "
        "while keeping the client's core issue and emotional experience essentially the same.\n\n"
        f"CURRENT text (~{target_words} words):\n{current}\n"
        + reference_block + "\n"
        f"FINAL PROFILE ATTRIBUTES (the result must stay coherent with ALL of these; "
        f"do NOT insert them into the text unless already present):\n"
        + "\n".join(f"  {k}: {v}" for k, v in final_attrs.items()) + "\n\n"
        "SYMPTOMS TO KEEP CONSISTENT (do NOT quote severity scores):\n"
        + "\n".join(symptom_lines) + "\n\n"
        "Revision rules:\n"
        "  - Keep the client's CORE issue, central distress, and its severity exactly the same\n"
        "  - Peripheral details may be swapped only for very close equivalents (e.g. interest in\n"
        "    music -> interest in dance, a sibling -> a cousin) — never for something distant or unrelated\n"
        "  - If using the REFERENCE, take at most ONE small detail or just its style — never its core issue\n"
        "  - Every substitution must stay coherent with the FINAL PROFILE ATTRIBUTES above\n"
        f"  - Keep the SAME length, about {target_words} words — do NOT make it longer\n"
        "  - Keep third-person voice; do NOT introduce demographic descriptors not already in the text\n"
        "  - Do NOT include numerical severity scores\n"
        "  - Vary vocabulary and sentence structure from the current text\n"
        "  - Return ONLY the revised text, no preamble."
    )
    return call_vllm([{"role": "user", "content": prompt}], max_tokens=350, temperature=0.9)


def build_situation(orig: dict, new: dict, source_idx: int) -> tuple[str, dict]:
    original_situation = orig.get("situation of the client", "")
    target_words = len(original_situation.split())

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
    current = adapt_situation(orig, new, target_words)
    current = enforce_length(current, target_words)
    sim = cosine_similarity(original_situation, current)

    iters = 0
    best = current
    best_sim = sim
    while sim > args.sim_threshold and iters < args.max_iters:
        reference = retrieve_reference(new, source_idx)
        candidate = diverge_situation(current, final_attrs, symptom_lines,
                                      target_words, reference)
        candidate = enforce_length(candidate, target_words)
        cand_sim = cosine_similarity(original_situation, candidate)
        iters += 1

        if cand_sim < best_sim:
            best, best_sim = candidate, cand_sim

        current, sim = candidate, cand_sim

    info = {
        "iters": iters,
        "final_sim": best_sim,
        "reached_threshold": best_sim <= args.sim_threshold,
        "target_words": target_words,
        "final_words": len(best.split()),
    }
    return best, info

df_out = df.copy()
n_modified  = 0
n_unchanged = 0
n_errors    = 0
n_reached   = 0   
sim_accum    = 0.0
iter_accum   = 0
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


        new_situation, info = build_situation(orig, new, idx)
        new["situation of the client"] = new_situation
        sim_accum  += info["final_sim"]
        iter_accum += info["iters"]
        if info["reached_threshold"]:
            n_reached += 1

        for sf in SYMPTOM_FIELDS | FIXED_FIELDS:
            if sf in orig and orig[sf] != new.get(sf):
                raise ValueError(f"BUG: field '{sf}' was altered at idx={idx}!")

        df_out.at[idx, "profile"] = json.dumps(new, indent=1)
        n_modified += 1

        if (loop_i + 1) % 10 == 0 or loop_i < 5:
            print(f"  [{loop_i+1:3d}/{len(sorted_valid)}] idx={idx:4d}  "
                    f"changed={changed_attrs}  sim={info['final_sim']:.3f}  iters={info['iters']}  "
                    f"words={info['final_words']}/{info['target_words']}")

    except Exception as e:
        n_errors += 1
        print(f"  [{loop_i+1:3d}/{len(sorted_valid)}] idx={idx:4d}  ERROR: {e}")
        traceback.print_exc()
df_out.to_parquet(args.output, index=False)

avg_sim  = (sim_accum  / n_modified) if (n_modified) else float("nan")
avg_iter = (iter_accum / n_modified) if (n_modified) else float("nan")

print(f"""
{'='*52}
  Summary
{'='*52}
  Valid indices processed   : {len(sorted_valid)}
  Profiles modified         : {n_modified}
  Profiles unchanged        : {n_unchanged}
  Reached sim threshold     : {n_reached} / {n_modified}
  Avg final similarity      : {avg_sim:.3f}
  Avg divergence iterations : {avg_iter:.2f}
  Errors                    : {n_errors}
  Output                    : {args.output}
{'='*52}
""")
