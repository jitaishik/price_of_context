import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
from langchain.prompts import PromptTemplate

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.json_retry import call_llm_json
from common.llm_client import get_vllm_client_for_model
from common.profile_attributes import get_client_information
from common.prompt_utils import load_prompt

REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE_GEN_PROMPTS_DIR = Path(__file__).resolve().parent / "profile_gen_prompts"
EEYORE_DATA_PATH = Path(__file__).resolve().parents[1] / "eeyore-data.parquet"
VALID_INDICES_PATH = REPO_ROOT / "valid_indices.json"


def get_symptoms_profiles(client, model_url, symptoms_s):
    system_prompt = load_prompt(PROFILE_GEN_PROMPTS_DIR, "symptoms_system.txt")
    user_prompt_text = load_prompt(PROFILE_GEN_PROMPTS_DIR, "symptoms_user.txt")
    prompt_template = PromptTemplate(
        input_variables=["symptoms"],
        template=user_prompt_text)
    user_prompt = prompt_template.format(symptoms=symptoms_s)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return call_llm_json(
        client, model_url, messages,
        temperature=0.7, max_tokens=500, max_retries=5,
        strip_code_fence="backtick")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Synthetic Profile.")
    parser.add_argument("-o", "--output_dir", type=str, default=".",
                        help="Directory to save the session results.")
    parser.add_argument("-model", "--model_name", type=str, default="llama",
                        help="Name of the model used for situation generation.")
    args = parser.parse_args()
    
    model_url, client = get_vllm_client_for_model(args.model_name)

    output_dir_name = args.output_dir+"_"+args.model_name
    output_dir = Path(output_dir_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(VALID_INDICES_PATH, "r") as f:
        valid_indices = json.load(f)

    source_list = []

    df = pd.read_parquet(EEYORE_DATA_PATH)
    profile_list = df["profile"].tolist()
    id_list = df["id_source"].tolist()
    for i in range(len(profile_list)):
        if id_list[i] not in source_list and i in valid_indices:
            source_list.append(id_list[i])
            if f"session_{i+1}.json" in os.listdir(output_dir_name):
                continue
            profile_i = json.loads(profile_list[i])
            _, client_medical_information_i, client_resistance_i, _ = get_client_information(profile_i)
            output_symptoms = get_symptoms_profiles(client, model_url, client_medical_information_i)
            client_background_information = ""
            client_background_information += "name" + ": " + output_symptoms["name"] + "\n"
            if output_symptoms["gender"] != "Cannot be identified":
                client_background_information += "gender" + ": " + output_symptoms["gender"] + "\n"
            if output_symptoms["age"] != "Cannot be identified":
                client_background_information += "age" + ": " + output_symptoms["age"] + "\n"
            if output_symptoms["occupation"] != "Cannot be identified":
                client_background_information += "occupation" + ": " + output_symptoms["occupation"] + "\n"
            if output_symptoms["marital_status"] != "Cannot be identified":
                client_background_information += "marital status" + ": " + output_symptoms["marital_status"] + "\n"
            client_background_information += "situation of the client" + ": " + output_symptoms["situation"] + "\n"
            profile_dict_i = {
                "client_background_information": client_background_information,
                "client_medical_information": client_medical_information_i,
                "client_resistance": client_resistance_i,
            }
            with open(os.path.join(output_dir_name, f"session_{i+1}.json"), "w") as f:
                json.dump(profile_dict_i, f, indent=4)
