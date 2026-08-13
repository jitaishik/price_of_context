import argparse
import json
import os
import sys
from pathlib import Path

import openai
from langchain.prompts import PromptTemplate

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.json_retry import call_with_backoff
from common.llm_client import configure_gpt, get_vllm_client_for_model
from common.prompt_utils import load_prompt, save_as_json

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
NUM_ITEMS = 12


def generate_history(history):
    return "\n".join(
        f"{message['role'].capitalize()}: {message['content']}"
        for message in history
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Score synthetic counseling sessions on the Working Alliance Inventory (WAI).")
    parser.add_argument("-i", "--input_dir", type=str, default=".",
                        help="Directory to read the sessions")
    parser.add_argument("-o", "--output_dir", type=str, default=".",
                        help="Directory to to save the results.")
    parser.add_argument("-m_iter", "--max_iter", type=int, default=3,
                        help="Maximum number of turns for the session.")
    parser.add_argument("-model", "--model_name", type=str, default="gpt",
                        help="Judge Model Name. Choose from llama, qwen, gpt and gpt-oss")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    configure_gpt()
    model_url, client = (None, None)
    if args.model_name != "gpt":
        model_url, client = get_vllm_client_for_model(args.model_name)

    for filename in os.listdir(args.input_dir):
        if filename.endswith(".json") and filename not in os.listdir(args.output_dir):
            file_path = os.path.join(args.input_dir, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)
            score_dict = {}
            for i in range(NUM_ITEMS):
                prompt_text = load_prompt(PROMPTS_DIR, f"wai{i+1}.txt")
                prompt_template = PromptTemplate(
                    input_variables=["conversation"],
                    template=prompt_text)
                prompt = prompt_template.format(conversation=generate_history(json_data["history"]))
                messages = [{"role": "user", "content": prompt}]
                if args.model_name == "gpt":
                    response = call_with_backoff(lambda: openai.chat.completions.create(
                        model="gpt-4o",
                        messages=messages,
                        temperature=0,
                        n=args.max_iter), max_retries=5)
                else:
                    response = call_with_backoff(lambda: client.chat.completions.create(
                        model=model_url,
                        messages=messages,
                        temperature=0,
                        n=args.max_iter), max_retries=5)
                score = 0
                for j in range(args.max_iter):
                    txt_response = response.choices[j].message.content
                    score = score + int(txt_response.split(",")[0])
                score_dict[f"wai{i+1}"] = score / args.max_iter
            save_as_json(score_dict, os.path.join(args.output_dir, filename))
