# Evaluation / WAI

Scores generated counseling sessions on the **Working Alliance Inventory (WAI)**, using an LLM-as-a-judge over twelve items (1-7 each) spanning the Task, Goal, and Bond components of therapeutic alliance.

| Argument | Description |
|---|---|
| `-i` / `--input_dir` | Directory of session JSON files to score |
| `-o` / `--output_dir` | Directory to write per-session score JSON files |
| `-m_iter` / `--max_iter` | Number of judge samples averaged per item (default `3`) |
| `-model` / `--model_name` | Judge model: `llama`, `qwen`, `gpt`, or `gpt-oss` (default `gpt`) |

## Items

`prompts/wai1.txt` through `prompts/wai12.txt`, one item prompt each.

```bash
python wai.py -i sessions/ -o wai_results/ -model gpt
```

## Aggregating Results

`rating.ipynb` reads a directory of `wai.py` output files and, for each of the twelve items, averages the score across all valid sessions (per `../../valid_indices.json`).
