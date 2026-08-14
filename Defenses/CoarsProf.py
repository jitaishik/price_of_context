import argparse
import json
import sys
import time
from pathlib import Path

import openai
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.llm_client import get_vllm_client_for_model

REPO_ROOT = Path(__file__).resolve().parents[1]
EEYORE_DATA_PATH = REPO_ROOT / "eeyore-data.parquet"
VALID_INDICES_PATH = REPO_ROOT / "valid_indices.json"

# ---- occupation → broad category ----------------------------------------
OCCUPATION_CATEGORIES = {
    # cluster 0 -> Student
    'Academic Research Associate': 'Student',
    'PhD in immunology': 'Student',
    'PhD student': 'Student',
    'Medical student': 'Student',
    "Graduate in a field related to their master's degree": 'Student',
    'Recent college graduate looking for a job in economic research': 'Student',
    'Film Student and Podcast Host': 'Student',
    'Student': 'Student',
    'College student': 'Student',
    'college student': 'Student',
    'Intern': 'Student',
    # cluster 1 -> Technology
    'Computer Science Engineer': 'Technology',
    'Computer stuff': 'Technology',
    'Electrical engineer': 'Technology',
    'Engineer in Mechanical Engineering': 'Technology',
    'Tech person': 'Technology',
    'Technical support specialist': 'Technology',
    'Customer service and technical expert': 'Technology',
    # cluster 2 -> Healthcare
    'Doctor': 'Healthcare',
    'Retired Nurse': 'Healthcare',
    # cluster 3 -> Education
    'Special education teacher': 'Education',
    'Assistant religious ed teacher and daycare worker': 'Education',
    # cluster 4 -> Business/Finance
    'Accounting. Low entry, bookkeeper.': 'Business/Finance',
    'Bank employee': 'Business/Finance',
    'Teller at a bank': 'Business/Finance',
    'Employee in a company, potentially managerial or administrative': 'Business/Finance',
    'Commission based job': 'Business/Finance',
    # cluster 5 -> Service
    'Retail': 'Service',
    'Home Depot employee': 'Service',
    'Kmart employee': 'Service',
    "McDonald's employee": 'Service',
    'Gas station worker': 'Service',
    'Call center employee': 'Service',
    'Hotel or Conference Center Worker': 'Service',
    'Security worker': 'Service',
    'hair braider': 'Service',
    'Cleaning job': 'Service',
    # cluster 6 -> Creative/Arts
    'Writer': 'Creative/Arts',
    'Drummer': 'Creative/Arts',
    'Music industry professional': 'Creative/Arts',
    # cluster 7 -> Other
    'Military': 'Other',
    'Peace Corps volunteer': 'Other',
    'Non-profit library system employee': 'Other',
    'Self employed': 'Other',
    'Works from home': 'Other',
    'Part-time job': 'Other',
    # cluster 8 -> Unemployed
    'Unemployed': 'Unemployed',
}


def generalise_occupation(raw: str) -> str:
    """Map a free-text occupation to a broad category."""
    if not raw or raw.strip().lower() in ("", "n/a", "none", "unknown"):
        return "Occupation Not Specified"
    lower = raw.lower()
    for keyword, category in OCCUPATION_CATEGORIES.items():
        if keyword in lower:
            return category
    return "Employed (Other)"

MARITAL_MAP = {
    'Currently experiencing a breakup with her husband.': 'Separated',
    'Divorced': 'Divorced',
    'Engaged': 'In a relationship',
    'In a relationship': 'In a relationship',
    'In a relationship, unmarried': 'In a relationship',
    'Married': 'Married',
    'Not married': 'Other',
    'Separated': 'Separated',
    'Single': 'Single',
    'single': 'Single',
    'Unmarried': 'Other',
    'Widowed': 'Widowed',
    'widowed': 'Widowed',
}


def generalise_marital_status(raw: str) -> str:
    if not raw or raw.strip().lower() in ("", "n/a", "none", "unknown"):
        return "Marital Status Not Specified"
    lower = raw.lower()
    for keyword, category in MARITAL_MAP.items():
        if keyword in lower:
            return category
    return "Marital Status Not Specified"

DROP_FIELDS = {"name", "ethnicity", "nationality", "race", "religion"}


SITUATION_SYSTEM_PROMPT = """You are a careful text editor working on a mental health research dataset.
 
STRICT RULE: Make only the minimal edits necessary to remove or generalise identifying information. Do NOT rephrase, restructure, or rewrite sentences that do not contain identifying information. Preserve all original wording that does not need to change.
 
Your task: make targeted edits to the "situation of the client" description so that it:
1. Removes or generalises potentially identifying information:
   - Personal names → replace with "the client", "a family member", "a friend", etc.
   - Named institutions (schools, companies, hospitals, cities, countries) → replace with generic equivalents like "a university", "a workplace", "a hospital", "a city"
   - Unique personal details that could identify a specific person (e.g. very specific job titles, rare diagnoses combined with unusual life events) → generalise minimally
   If none of these are present, return the original text unchanged.
2. Preserves all clinically and emotionally relevant content:
   - The nature of the psychological distress
   - Relationship dynamics and interpersonal conflicts
   - Life circumstances that are contextually important (job loss, bereavement, academic pressure, etc.)
   - The severity and duration of symptoms
3. Keeps the description in third-person, in the same length and style as the original.
 
Return ONLY the edited text, with no preamble, explanation, or quotation marks."""


def generalise_situation_llm(
    client: openai.OpenAI, model: str, situation: str
) -> str:
    """Call the local Qwen/vLLM server to anonymize a single situation string."""
    if not situation or situation.strip() == "":
        return situation
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SITUATION_SYSTEM_PROMPT},
            {"role": "user", "content": situation.strip()},
        ],
        temperature=0.7, 
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()


def generalise_situation_batch(
    client: openai.OpenAI, model: str, situations: list[str]
) -> list[str]:
    """Generalise a list of situation strings, one call each, with retry."""
    results = []
    for sit in tqdm(situations, desc="  Anonymising situations", leave=False):
        for attempt in range(3):
            try:
                results.append(generalise_situation_llm(client, model, sit))
                break
            except openai.RateLimitError:
                wait = 2 ** attempt * 5
                print(f"    Rate limit hit, waiting {wait}s …")
                time.sleep(wait)
            except Exception as e:
                if attempt == 2:
                    print(f"Warning: could not anonymise situation after 3 attempts: {e}")
                    results.append(sit)  
                time.sleep(2)
    return results

def anonymize_profile(profile: dict, anon_situation: str | None = None) -> dict:
    new_profile = {}
    for key, value in profile.items():
        lower_key = key.lower().strip()

        if lower_key in DROP_FIELDS:
            continue

        if lower_key == "occupation":
            new_profile[key] = generalise_occupation(str(value))
            continue

        if lower_key == "marital status":
            new_profile[key] = generalise_marital_status(str(value))
            continue

        if lower_key == "situation of the client" and anon_situation is not None:
            new_profile[key] = anon_situation
            continue

        new_profile[key] = value

    return new_profile

def main():
    parser = argparse.ArgumentParser(description="Anonymize eeyore profile data.")
    parser.add_argument("-o", "--output", default="eeyore-data-anon.parquet",
                        help="Path to write the anonymized parquet file.")
    parser.add_argument("-model", "--model_name", type=str, default="llama",
                        help="Name of the model used for generation.")
    parser.add_argument("--batch-size", type=int, default=50,
                        help="Number of rows to process per progress checkpoint.")
    parser.add_argument("--no-llm", action="store_true",
                        help="Skip LLM-based situation anonymization (only apply rule-based field generalisation).")
    args = parser.parse_args()

    # Load data, restricted to valid indices only
    print(f"Loading {EEYORE_DATA_PATH} …")
    df_full = pd.read_parquet(EEYORE_DATA_PATH)

    with open(VALID_INDICES_PATH, "r") as f:
        valid_indices = json.load(f)
    valid = sorted(set(int(i) for i in valid_indices))

    df = df_full.iloc[valid].reset_index(drop=True)
    total = len(df)
    print(f"  {total} rows loaded ({len(df_full)} total, filtered to valid_indices).")

    output_path = Path(args.output)
    if output_path.exists():
        df_out = pd.read_parquet(args.output)
        start_idx = len(df_out)
        print(f"  Resuming from row {start_idx} (output file already has {start_idx} rows).")
    else:
        df_out = df.iloc[:0].copy() 
        start_idx = 0

    if start_idx >= total:
        print("Already complete. Nothing to do.")
        return

    vllm_client = None
    model_url = None
    if not args.no_llm:
        model_url, vllm_client = get_vllm_client_for_model(args.model_name)
        print(f"  Connecting to vLLM server (model: {args.model_name}) …")

    rows_to_process = df.iloc[start_idx:].reset_index(drop=True)
    new_rows = []

    for batch_start in tqdm(
        range(0, len(rows_to_process), args.batch_size),
        desc="Batches",
        unit="batch",
    ):
        batch = rows_to_process.iloc[batch_start : batch_start + args.batch_size].copy()

        profiles = [json.loads(p) for p in batch["profile"]]

        if vllm_client is not None:
            situations = [p.get("situation of the client", "") for p in profiles]
            anon_situations = generalise_situation_batch(vllm_client, model_url, situations)
        else:
            anon_situations = [None] * len(profiles)

        anon_profiles = [
            anonymize_profile(p, anon_sit)
            for p, anon_sit in zip(profiles, anon_situations)
        ]
        batch["profile"] = [json.dumps(ap) for ap in anon_profiles]
        new_rows.append(batch)

        checkpoint = pd.concat([df_out] + new_rows, ignore_index=True)
        checkpoint.to_parquet(args.output, index=False)

    print(f"\nDone. Anonymized data written to {args.output}")
    print(f"  Total rows: {len(pd.read_parquet(args.output))}")


if __name__ == "__main__":
    main()
