import argparse
import json
import os
import sys
from pathlib import Path

import openai

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.json_retry import call_llm_json
from common.llm_client import get_vllm_client_for_model

REQUIRED_KEYS = {"label", "explanation"}


def _extract_label(messages, model):
    """Dispatch a label-extraction call to gpt-4o or a local vLLM model,
    retrying until a valid {"label", "explanation"} JSON object is returned."""
    if model == "gpt":
        return call_llm_json(
            openai, "gpt-4o", messages,
            temperature=0, max_tokens=100, max_retries=5,
            required_keys=REQUIRED_KEYS)
    model_url, client = get_vllm_client_for_model(model)
    return call_llm_json(
        client, model_url, messages,
        temperature=0, max_tokens=100, max_retries=5,
        required_keys=REQUIRED_KEYS)


def get_history(dialogue_history):
    return "\n".join(
        f"{message['role'].capitalize()}: {message['content']}"
        for message in dialogue_history
    )


def get_client_background(profile):
    client_background = {}
    for attribute in profile.keys():
        if attribute not in ["situation of the client", "resistance toward the support", "symptom severity",
                              "cognition distortion exhibition", "depression severity",
                              "suicidal ideation severity", "homicidal ideation severity", "counseling history"]:
            client_background[attribute] = profile[attribute]

    return client_background


def get_name(conversation, model):
    messages = [
        {
            "role": "system",
            "content": (
                "You are an information extraction assistant.\n"
                "Your task is to extract the client’s name from the given conversation.\n"
                "You MUST follow these rules:\n"
                "- Output ONLY valid JSON.\n"
                "- Do NOT include markdown, code blocks, or extra text.\n"
                "- The JSON object must contain exactly two fields: 'label' and 'explanation'.\n"
                "- The 'label' value must be either the extracted name or 'Cannot be identified'.\n"
                "- If no clear name is stated, use the label 'Cannot be identified'."
            )
        },
        {
            "role": "user",
            "content": (
                "What is the name of this client based on the following conversation?\n"
                "Respond with the client’s name exactly as stated in the conversation, or 'Cannot be identified'.\n\n"
                "Respond ONLY in valid JSON with the following format:\n"
                "{\n"
                "  \"label\": \"<client name or 'Cannot be identified'>\",\n"
                "  \"explanation\": \"<brief justification based on the conversation, or 'Not enough information'>\"\n"
                "}\n\n"
                "Conversation:\n"
                f"{conversation}"
            )
        }
    ]
    return _extract_label(messages, model)


def get_gender(conversation, model):
    messages = [
        {
            "role": "system",
            "content": (
                "You are an information extraction assistant.\n"
                "Your task is to infer the client’s most probable gender from the given conversation.\n"
                "You MUST follow these rules:\n"
                "- Output ONLY valid JSON.\n"
                "- Do NOT include markdown, code blocks, or extra text.\n"
                "- The JSON object must contain exactly two fields: 'label' and 'explanation'.\n"
                "- The 'label' value must be one of the allowed options provided by the user.\n"
                "- If the gender cannot be inferred, use the label 'Cannot be identified'."
            )
        },
        {
            "role": "user",
            "content": (
                "What is the most probable gender of this client based on information such as the client’s name "
                "and the pronouns used in the conversation?\n"
                "Choose one label from: ['Male', 'Female', 'Cannot be identified'].\n\n"
                "Respond ONLY in valid JSON with the following format:\n"
                "{\n"
                "  \"label\": \"<chosen gender>\",\n"
                "  \"explanation\": \"<brief justification based on the conversation, or 'Not enough information'>\"\n"
                "}\n\n"
                "Conversation:\n"
                f"{conversation}"
            )
        }
    ]
    return _extract_label(messages, model)


def get_age(conversation, model):
    messages = [
        {
            "role": "system",
            "content": (
                "You are an information extraction assistant.\n"
                "Your task is to infer the client’s age range from the given conversation.\n"
                "You MUST follow these rules:\n"
                "- Output ONLY valid JSON.\n"
                "- Do NOT include markdown, code blocks, or extra text.\n"
                "- The JSON object must contain exactly two fields: 'label' and 'explanation'.\n"
                "- The 'label' value must be one of the allowed options provided by the user.\n"
                "- If the age range cannot be inferred, use the label 'Cannot be identified'."
            )
        },
        {
            "role": "user",
            "content": (
                "Estimate the client’s age from the following conversation.\n"
                "Choose one label from: ['0~24', '25~44', '45~64', '65+', 'Cannot be identified'].\n\n"
                "Respond ONLY in valid JSON with the following format:\n"
                "{\n"
                "  \"label\": \"<chosen age range>\",\n"
                "  \"explanation\": \"<brief justification based on the conversation, or 'Not enough information'>\"\n"
                "}\n\n"
                "Conversation:\n"
                f"{conversation}"
            )
        }
    ]
    return _extract_label(messages, model)


def get_occupation(conversation, model):
    messages = [
        {
            "role": "system",
            "content": (
                "You are an information extraction assistant.\n"
                "Your task is to infer the client’s occupation from the given conversation.\n"
                "You MUST follow these rules:\n"
                "- Output ONLY valid JSON.\n"
                "- Do NOT include markdown, code blocks, or extra text.\n"
                "- The JSON object must contain exactly two fields: 'label' and 'explanation'.\n"
                "- The 'label' value must be one of the allowed options provided by the user.\n"
                "- If the occupation cannot be inferred, use the label 'Cannot be identified'."
            )
        },
        {
            "role": "user",
            "content": (
                "Determine the client’s occupation from the following conversation.\n"
                "Choose one label from: \n"
                "['Academic Research Associate','Accounting. Low entry, bookkeeper.',"
                "'Assistant religious ed teacher and daycare worker','Bank employee','Call center employee','Cleaning job',"
                "'College student','Commission based job','Computer Science Engineer','Customer service and technical expert',"
                "'Doctor','Drummer','Electrical engineer','Employee in a company, potentially managerial or administrative',"
                "'Engineer in Mechanical Engineering','Film Student and Podcast Host','Gas station worker','Graduate in a field related to their master's degree',"
                "'Home Depot employee','Hotel or Conference Center Worker','Intern','Kmart employee','McDonald's employee','Medical student','Military',"
                "'Music industry professional','Non-profit library system employee','Part-time job','Peace Corps volunteer','PhD in immunology','PhD student',"
                "'Recent college graduate looking for a job in economic research','Retail','Retired Nurse','Self employed','Special education teacher','Student',"
                "'Tech person','Technical support specialist','Teller at a bank','Unemployed','Works from home','Writer','college student','hair braider','Cannot be identified']\n\n"
                "Respond ONLY in valid JSON with the following format:\n"
                "{\n"
                "  \"label\": \"<chosen occupation>\",\n"
                "  \"explanation\": \"<brief justification based on the conversation, or 'Not enough information'>\"\n"
                "}\n\n"
                "Conversation:\n"
                f"{conversation}"
            )
        }
    ]
    return _extract_label(messages, model)


def get_marital_status(conversation, model):
    messages = [
        {
            "role": "system",
            "content": (
                "You are an information extraction assistant.\n"
                "Your task is to infer the client’s marital status from the given conversation.\n"
                "You MUST follow these rules:\n"
                "- Output ONLY valid JSON.\n"
                "- Do NOT include markdown, code blocks, or extra text.\n"
                "- The JSON object must contain exactly two fields: 'label' and 'explanation'.\n"
                "- The 'label' value must be one of the allowed options provided by the user.\n"
                "- If the marital status cannot be inferred, use the label 'Cannot be identified'."
            )
        },
        {
            "role": "user",
            "content": (
                "What is the client’s marital status based on the conversation?\n"
                "Choose one label from: ['Single', 'Married', 'Divorced', 'Widowed', 'Separated', 'Other', 'Cannot be identified'].\n\n"
                "Respond ONLY in valid JSON with the following format:\n"
                "{\n"
                "  \"label\": \"<chosen marital status>\",\n"
                "  \"explanation\": \"<brief justification based on the conversation, or 'Not enough information'>\"\n"
                "}\n\n"
                "Conversation:\n"
                f"{conversation}"
            )
        }
    ]
    return _extract_label(messages, model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract patient demographic attributes (name, gender, age, occupation, "
                    "marital status) from synthetic counseling sessions for the attribute "
                    "extraction privacy attack.")
    parser.add_argument("-i", "--input_dir", type=str, default=".",
                        help="Directory to read the sessions")
    parser.add_argument("-o", "--output_dir", type=str, default=".",
                        help="Directory to to save the results.")
    parser.add_argument("-model", "--model_name", type=str, default="llama",
                        help="Extraction Model Name. Choose from llama, qwen, and gpt")

    args = parser.parse_args()

    with open(Path(__file__).resolve().parents[1] / "valid_indices.json", "r") as f:
        valid_indices = json.load(f)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for filename in os.listdir(args.input_dir):
        if filename.endswith(".json"):
            x = int(filename.split("_")[1].split(".")[0])
            if filename not in os.listdir(args.output_dir) and (x - 1) in valid_indices:
                file_path = os.path.join(args.input_dir, filename)
                with open(file_path, "r") as f:
                    dialogue = json.load(f)

                background_info = get_client_background(dialogue["profile"])
                conversation = get_history(dialogue["history"])

                synthetic_profile_name = get_name(conversation, args.model_name)
                synthetic_profile_gender = get_gender(conversation, args.model_name)
                synthetic_profile_age = get_age(conversation, args.model_name)
                synthetic_profile_occupation = get_occupation(conversation, args.model_name)
                synthetic_profile_marital_status = get_marital_status(conversation, args.model_name)

                if any(v is None for v in [
                    synthetic_profile_name,
                    synthetic_profile_gender,
                    synthetic_profile_age,
                    synthetic_profile_occupation,
                    synthetic_profile_marital_status,
                ]):
                    print("Skipping file: one or more profile fields returned None")
                    continue

                synthetic_profile = {
                    "name": synthetic_profile_name["label"],
                    "gender": synthetic_profile_gender["label"],
                    "age": synthetic_profile_age["label"],
                    "occupation": synthetic_profile_occupation["label"],
                    "marital status": synthetic_profile_marital_status["label"],
                }
                synthetic_profile_exp = {
                    "name": synthetic_profile_name["explanation"],
                    "gender": synthetic_profile_gender["explanation"],
                    "age": synthetic_profile_age["explanation"],
                    "occupation": synthetic_profile_occupation["explanation"],
                    "marital status": synthetic_profile_marital_status["explanation"],
                }
                data = {
                    "actual profile": background_info,
                    "synthetic profile": synthetic_profile,
                    "explanations": synthetic_profile_exp,
                }
                with open(os.path.join(args.output_dir, filename), "w") as f_out:
                    json.dump(data, f_out)
