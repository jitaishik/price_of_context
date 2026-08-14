import glob
import json
import os
import re
import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.llm_client import get_vllm_client_for_model

REPO_ROOT = Path(__file__).resolve().parents[1]
VALID_INDICES_PATH = REPO_ROOT / "valid_indices.json"
MAX_TOKENS = 8192
TEMPERATURE = 0.0

SYSTEM_PROMPT = """You are a clinical data de-identification specialist.
Your task is to anonymize conversation transcripts by replacing personally identifiable information (PII) with neutral placeholders.

PII to detect and replace:
- Names (people, places that identify a specific person) → [NAME]
- Explicit ages or birth years → [AGE]
- Specific job titles or employer names → [OCCUPATION]
- Marital status (e.g., married, divorced, widowed, single) → [MARITAL_STATUS]
- Explicit gender pronouns or gender statements when the speaker explicitly states their gender → [GENDER]
- Specific company/organization names that could identify a person → [ORGANIZATION]
- Phone numbers, email addresses, addresses → [CONTACT_INFO]

Rules:
- Do NOT change anything else. Preserve the full meaning, emotional tone, and structure of each utterance.
- Do NOT alter generic references (e.g., "my family", "my boss", "a colleague") — only replace if they are specific and identifying.
- Do NOT add commentary or explanation.
- Return the transcript in the exact same format as the input: one turn per line, each prefixed with the speaker role (e.g. "Counselor: ..." / "Client: ...").
- If no PII is found, return the transcript unchanged.
"""

USER_TEMPLATE = """Anonymize the following counseling session transcript. Return only the transcript lines, no extra text.

{history_text}"""


def session_index(filename: str):
    match = re.search(r"session_(\d+)\.json$", os.path.basename(filename))
    return int(match.group(1)) if match else None


def get_history(dialogue_history: list) -> str:
    """Convert list of role/content dicts to a plain-text transcript."""
    return "\n".join(
        f"{message['role'].capitalize()}: {message['content']}"
        for message in dialogue_history
    )


def parse_history(history_text: str, original: list) -> list:
    original_roles = [m["role"] for m in original]
    result = []
    lines = [l for l in history_text.splitlines() if l.strip()]

    for i, line in enumerate(lines):
        if ":" not in line:
            raise ValueError(f"Cannot parse transcript line {i}: {line!r}")
        role_raw, content = line.split(":", 1)
        # Restore the original casing of the role from the source file
        role = original_roles[i] if i < len(original_roles) else role_raw.strip().lower()
        result.append({"role": role, "content": content.strip()})

    if len(result) != len(original):
        raise ValueError(
            f"Turn count mismatch: expected {len(original)}, got {len(result)}"
        )
    return result


def anonymize_history(client, model: str, history: list) -> list:
    """Send history to the judge model via vLLM and return anonymized version."""
    history_text = get_history(history)

    response = client.chat.completions.create(
        model=model,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(history_text=history_text)},
        ],
    )

    raw = response.choices[0].message.content.strip()
    return parse_history(raw, history)


def process_file(client, model: str, input_path: str, output_dir: str) -> str:
    """Process a single session file. Returns output path."""
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "history" not in data:
        raise ValueError(f"No 'history' key in {input_path}")

    anonymized_history = anonymize_history(client, model, data["history"])
    output_data = {
        key: (anonymized_history if key == "history" else value)
        for key, value in data.items()
    }

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, os.path.basename(input_path))
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Anonymize counseling session JSON files (NER-based de-identification defense)"
    )
    parser.add_argument(
        "input",
        nargs="+",
        help="Input JSON file(s) or glob pattern (e.g., './sessions/*.json')",
    )
    parser.add_argument(
        "-m", "--model_name",
        type=str,
        default="llama",
        help="Name of the model used for generation.",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default="./anonymized",
        help="Directory to write anonymized files (default: ./anonymized)",
    )
    parser.add_argument(
        "-d", "--delay",
        type=float,
        default=0.0,
        help="Seconds to wait between API calls to avoid overloading server (default: 0.0)",
    )
    args = parser.parse_args()

    files = []
    for pattern in args.input:
        matched = glob.glob(pattern)
        files.extend(matched if matched else ([pattern] if os.path.isfile(pattern) else []))

    if not files:
        print("No input files found.", file=sys.stderr)
        sys.exit(1)

    files = sorted(set(files))

    # Filter by valid indices: keep 'session_i.json' only if (i - 1) is valid.
    with open(VALID_INDICES_PATH, "r") as f:
        valid_indices = json.load(f)
    valid = set(int(i) for i in valid_indices)

    kept, skipped = [], []
    for path in files:
        idx = session_index(path)
        if idx is None:
            skipped.append((path, "filename does not match session_<i>.json"))
        elif (idx - 1) in valid:
            kept.append(path)
        else:
            skipped.append((path, f"index {idx - 1} not in valid_indices"))

    print(f"Valid idx  : {VALID_INDICES_PATH}  "
          f"({len(kept)} kept, {len(skipped)} skipped)")
    if skipped:
        for path, reason in skipped:
            print(f"  skip {os.path.basename(path)}: {reason}")

    files = kept
    if not files:
        print("No files remain after valid-indices filtering.", file=sys.stderr)
        sys.exit(1)

    print(f"Model      : {args.model_name}")
    print(f"Files      : {len(files)}  ->  output: {args.output_dir}\n")

    model_url, client = get_vllm_client_for_model(args.model_name)

    success, failed = 0, []
    for i, path in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {path} ...", end=" ", flush=True)
        try:
            out = process_file(client, model_url, path, args.output_dir)
            print(f"OK  -> {out}")
            success += 1
        except Exception as e:
            print(f"FAILED: {e}")
            failed.append((path, str(e)))

        if args.delay and i < len(files):
            time.sleep(args.delay)

    print(f"\n{'─' * 60}")
    print(f"Done. {success} succeeded, {len(failed)} failed.")
    if failed:
        print("\nFailed files:")
        for path, err in failed:
            print(f"  {path}: {err}")


if __name__ == "__main__":
    main()
