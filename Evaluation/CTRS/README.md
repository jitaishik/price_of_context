# Evaluation / CTRS

Scores generated counseling sessions on the **Cognitive Therapy Rating Scale (CTRS)**, using an LLM-as-a-judge over six rubric dimensions (0-6 each): three general counseling skills and three CBT-specific skills.

| Argument | Description |
|---|---|
| `-i` / `--input_dir` | Directory of session JSON files to score |
| `-o` / `--output_dir` | Directory to write per-session score JSON files |
| `-m_iter` / `--max_iter` | Number of judge samples averaged per dimension (default `3`) |
| `-model` / `--model_name` | Judge model: `llama`, `qwen`, `gpt`, or `gpt-oss` (default `gpt`) |

## Rubrics

| Category | Dimension | Prompt file |
|---|---|---|
| General | Understanding | `prompts/general_1_understanding.txt` |
| General | Interpersonal Effectiveness | `prompts/general_2_interpersonal_effectiveness.txt` |
| General | Collaboration | `prompts/general_3_collaboration.txt` |
| CBT-specific | Guided Discovery | `prompts/CBT_1_guided_discovery.txt` |
| CBT-specific | Focus | `prompts/CBT_2_focus.txt` |
| CBT-specific | Strategy | `prompts/CBT_3_strategy.txt` |

```bash
python ctrs.py -i sessions/ -o ctrs_results/ -model gpt
```
