"""Generate Full Profile (FP) sessions from a *defended* profile parquet
file (produced by Defenses/generalize_profiles.py, create_eeyore_dp_simple.py,
or create_eeyore_dp_mix_iter.py). Identical generation logic to
Generation/script_profile.py; the only difference is that the input parquet
is required explicitly via --input_file rather than the raw eeyore-data.parquet."""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from langchain.prompts import PromptTemplate

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.json_retry import call_llm_json
from common.llm_client import get_vllm_client_for_model
from common.profile_attributes import get_client_information
from common.prompt_utils import fix_encoding_txt, load_prompt, save_as_json

REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPTS_DIR = REPO_ROOT / "prompts_script"
VALID_INDICES_PATH = REPO_ROOT / "valid_indices.json"

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
print(torch.cuda.get_device_capability()[0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Full Profile (FP) sessions from a defended profile parquet "
                     "(CoarsProf / NoiseProf / NoiseIterMix).")
    parser.add_argument("-o", "--output_dir", type=str, default=".",
                        help="Directory to save the session results.")
    parser.add_argument("-model", "--model_name", type=str, default="llama",
                        help="Name of the model used for generation.")
    parser.add_argument("-i", "--input_file", type=str, required=True,
                        help="Path to the defended profile parquet file, e.g. "
                             "eeyore-data-anon.parquet (CoarsProf), eeyore-dp-simple.parquet "
                             "(NoiseProf), or eeyore-dp-mix-iter.parquet (NoiseIterMix).")
    args = parser.parse_args()

    model_url, client = get_vllm_client_for_model(args.model_name)

    df = pd.read_parquet(args.input_file)
    profile_list = df["profile"].tolist()
    id_list = df["id_source"].tolist()

    with open(VALID_INDICES_PATH, "r") as f:
        valid_indices = json.load(f)

    source_list = []
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_text = load_prompt(PROMPTS_DIR, "profile.txt")

    for i in range(len(profile_list)):
        if len(source_list) == 500:
            break
        if id_list[i] not in source_list and i in valid_indices:
            source_list.append(id_list[i])
            if f"session_{i+1}.json" in os.listdir(args.output_dir):
                continue

            profile_i = json.loads(profile_list[i])
            client_background_information_i, client_medical_information_i, client_resistance_i, _ = (
                get_client_information(profile_i, mode="default"))

            prompt_template = PromptTemplate(
                input_variables=["client_background_information", "client_medical_information", "client_resistance"],
                template=prompt_text)
            prompt = prompt_template.format(
                client_background_information=client_background_information_i,
                client_medical_information=client_medical_information_i,
                client_resistance=client_resistance_i)

            messages = [{"role": "user", "content": prompt}]
            dialogue = call_llm_json(
                client, model_url, messages,
                temperature=0.7, max_retries=3,
                strip_code_fence="regex", decode_unicode_escape=True,
                raise_on_exhausted=True)

            d_cleaned = {"profile": profile_i, "history": []}
            for turn in dialogue:
                d_cleaned["history"].append({
                    "role": turn["role"],
                    "content": fix_encoding_txt(turn["content"]),
                })

            save_as_json(d_cleaned, os.path.join(args.output_dir, f"session_{i+1}.json"))
