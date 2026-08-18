import argparse
import json
import os
import random
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


def get_random_profiles(client, model_url, age_r, occupation_r, marital_status_r, gender_r, symptoms_r):
    system_prompt = load_prompt(PROFILE_GEN_PROMPTS_DIR, "random_system.txt")
    user_prompt_text = load_prompt(PROFILE_GEN_PROMPTS_DIR, "random_user.txt")
    prompt_template = PromptTemplate(
        input_variables=["age", "gender", "occupation", "marital_status", "symptoms"],
        template=user_prompt_text)
    user_prompt = prompt_template.format(
        age=age_r,
        gender=gender_r,
        occupation=occupation_r,
        marital_status=marital_status_r,
        symptoms=symptoms_r)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return call_llm_json(
        client, "model_path", messages,
        temperature=0.7, max_tokens=500, max_retries=5,
        required_keys={"name", "situation"}, strip_code_fence="backtick")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Random Profile.")
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
    random.seed(42)
    age_list = ['0~24', '25~44', '45~64', '65+', 'Cannot be identified']
    occupation_list = ["Academic Research Associate","Accounting. Low entry, bookkeeper.",
                     "Assistant religious ed teacher and daycare worker","Bank employee","Call center employee","Cleaning job",
                     "College student","Commission based job","Computer Science Engineer","Customer service and technical expert",
                     "Doctor","Drummer","Electrical engineer","Employee in a company, potentially managerial or administrative",
                     "Engineer in Mechanical Engineering","Film Student and Podcast Host","Gas station worker","Graduate in a field related to their master's degree",
                     "Home Depot employee","Hotel or Conference Center Worker","Intern","Kmart employee","McDonald's employee","Medical student","Military",
                     "Music industry professional","Non-profit library system employee","Part-time job","Peace Corps volunteer","PhD in immunology","PhD student",
                     "Recent college graduate looking for a job in economic research","Retail","Retired Nurse","Self employed","Special education teacher","Student",
                     "Tech person","Technical support specialist","Teller at a bank","Unemployed","Works from home","Writer","college student","hair braider","Cannot be identified"]
    marital_list = ['Single', 'Married', 'Divorced', 'Widowed', 'Separated', 'Other', 'Cannot be identified']
    gender_list = ['Male', 'Female', 'Cannot be identified']

    for i in range(len(profile_list)):
        if id_list[i] not in source_list and i in valid_indices:
            source_list.append(id_list[i])
            if f"session_{i+1}.json" in os.listdir(output_dir_name):
                continue
            profile_i = json.loads(profile_list[i])
            _, client_medical_information_i, client_resistance_i, _ = get_client_information(profile_i)
            age_r = random.choice(age_list)
            occupation_r = random.choice(occupation_list)
            marital_status_r = random.choice(marital_list)
            gender_r = random.choice(gender_list)
            output = get_random_profiles(client, model_url, age_r, occupation_r, marital_status_r, gender_r, client_medical_information_i)
            client_background_information = ""
            client_background_information += "name" + ": " + output["name"] + "\n"
            if gender_r != "Cannot be identified":
                client_background_information += "gender" + ": " + gender_r + "\n"
            if age_r != "Cannot be identified":
                client_background_information += "age" + ": " + age_r + "\n"
            if occupation_r != "Cannot be identified":
                client_background_information += "occupation" + ": " + occupation_r + "\n"
            if marital_status_r != "Cannot be identified":
                client_background_information += "marital status" + ": " + marital_status_r + "\n"
            client_background_information += "situation of the client" + ": " + output["situation"] + "\n"
            profile_dict_i = {
                "client_background_information": client_background_information,
                "client_medical_information": client_medical_information_i,
                "client_resistance": client_resistance_i,
            }

            with open(os.path.join(output_dir_name, f"session_{i+1}.json"), "w") as f:
                json.dump(profile_dict_i, f, indent=4)
                
