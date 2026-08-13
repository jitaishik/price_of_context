import json
import os

import chardet


def load_prompt(base_dir, filename):
    """Read a prompt template file from `base_dir`."""
    file_path = os.path.join(base_dir, filename)
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def save_as_json(dictionary, filename):
    """Save `dictionary` to `filename` as JSON."""
    with open(filename, "w") as file:
        json.dump(dictionary, file, indent=4)
    print(f"Dictionary saved to {filename}")


def fix_encoding_txt(txt):
    """Detect encoding and safely re-decode as UTF-8."""
    raw = txt.encode("utf-8")

    detected = chardet.detect(raw)
    enc = detected["encoding"] or "utf-8"

    try:
        text = raw.decode(enc)
    except Exception:
        text = raw.decode("utf-8", errors="replace")
    try:
        text_fixed = text.encode("latin1").decode("utf-8")
    except Exception:
        text_fixed = text

    lines = [line.lstrip() for line in text_fixed.splitlines()]
    return "\n".join(lines)
