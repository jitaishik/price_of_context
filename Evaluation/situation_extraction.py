import argparse
import json
import os
import sys
from pathlib import Path

import openai

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.json_retry import call_with_backoff
from common.llm_client import get_vllm_client_for_model

def _extract_situation(messages, model):
    """Dispatch a situation-extraction call to gpt-4o or a local vLLM model."""
    if model == "gpt":
        return call_with_backoff(lambda: openai.chat.completions.create(
            model="gpt-4o", messages=messages,
            temperature=0, max_tokens=100
        ).choices[0].message.content.strip())
    model_url, client = get_vllm_client_for_model(model)
    return call_with_backoff(lambda: client.chat.completions.create(
        model=model_url, messages=messages,
        temperature=0, max_tokens=100
    ).choices[0].message.content.strip())

def get_situation(conversation, model):
    messages = [
        {
            "role": "user",
            "content": (
                "Briefly summarize the client's situation **before help-seeking** that triggered this conversation. "
                "Use a few sentences. Focus only on the core situation.\n\n"
                "Examples:\n"
                "Example 1:\n"
                "The client is feeling lost and struggling with life's setbacks, unable to attend college due to financial difficulties and being held back by the lack of graduation papers. The client feels hopeless, especially after dropping out of college, and is unsure of how to move forward.\n\n"
                "Example 2:\n"
                "The client is distressed about a broken laptop which contained a lengthy story they were writing. The event that led to this damage was the client's father getting angry about the client's grades and slamming the laptop, potentially damaging it. The client is worried about retrieving the lost work and has expressed feelings of wanting to die. The client also mentioned family issues, feeling scared of their father's physical behavior, and concerns about the potential impact on their brother, who has Down Syndrome.\n\n"
                "Conversation:\n"
                f"{conversation}"
            )
        }
    ]

    return _extract_situation(messages,model)


def get_history(dialogue_history):
    return "\n".join(
        f"{message['role'].capitalize()}: {message['content']}"
        for message in dialogue_history
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract patient situations from synthetic counseling sessions "
                    "for the situation-leakage / linkage / membership-inference measurements.")
    parser.add_argument("-i", "--input_dir", type=str, default=".",
                        help="Directory to read the sessions")
    parser.add_argument("-o", "--output_file", type=str, default=".",
                        help="Directory to to save the results.")
    parser.add_argument("-o_dir", "--output_dir", type=str, default=".",
                        help="Directory to to save the results.")
    parser.add_argument("-model", "--model_name", type=str, default="llama",
                        help="Extraction Model Name. Choose from llama, qwen, and gpt")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    final_dict = {}
    for filename in os.listdir(args.input_dir):
        if filename.endswith(".json"):
            file_x = os.path.splitext(filename)[0]
            file_path = os.path.join(args.input_dir, filename)
            with open(file_path, "r") as f:
                dialogue = json.load(f)
            conversation = get_history(dialogue["history"])
            client_situation = get_situation(conversation, args.model_name)
            final_dict[file_x] = client_situation

    with open(os.path.join(args.output_dir, args.output_file), "w") as f:
        json.dump(final_dict, f, indent=4)
