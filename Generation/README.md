# Generation

Scripts for generating synthetic counseling sessions from the Eeyore seed profiles (`../eeyore-data.parquet`) at each of the three privacy levels, plus scripts for generating and using *synthetic* (non-real) patient profiles as an alternative to real ones (Appendix E of the paper).

All session-generation scripts share the following common arguments:

| Argument | Description |
|---|---|
| `-o` / `--output_dir` | Directory to save the generated `session_<i>.json` files |
| `-model` / `--model_name` | Judge/generator model: `llama` or `qwen` (served locally via vLLM at `http://localhost:8000/v1`, see [`common/llm_client.py`](../common/llm_client.py)) |

## Session Generation from Real Profiles

| Privacy level | Script | Prompt |
|---|---|---|
| Symptom-Only (SO) | `script_symptoms.py` | `symptoms.txt` |
| Contextual Situation (CS) | `script_situation.py` | `situation.txt` |
| Full Profile (FP) | `script_profile.py` | `profile.txt` |

```bash
python script_symptoms.py  -o out_so -model llama
python script_situation.py -o out_cs -model llama
python script_profile.py   -o out_fp -model llama
```

## Synthetic Profile Generation

Builds synthetic (non-real) patient profiles as JSON files, one per source Eeyore symptom set, for the Appendix E comparison of generation grounded vs. not grounded in real patient profiles. Run from this directory; output directories (`random_synthetic_profiles_qwen`, `symptoms_synthetic_profiles_qwen`) must exist before running.

| Variant | Script | Strategy |
|---|---|---|
| Random attributes | `random_synthetic_profiles.py` | Randomly samples age/occupation/marital status/gender, then asks the LLM to invent only a name + situation |
| Symptoms-grounded | `symptoms_synthetic_profiles.py` | Asks the LLM to invent all demographic attributes and a situation from symptoms alone |

```bash
mkdir -p random_synthetic_profiles_qwen && python random_synthetic_profiles.py
mkdir -p symptoms_synthetic_profiles_qwen && python symptoms_synthetic_profiles.py
```

## Session Generation from Synthetic Profiles

Generates Full Profile (FP) sessions from the synthetic profiles above instead of real Eeyore profiles.

| Argument | Description |
|---|---|
| `--profile_source_dir` | Directory of pre-built synthetic profile JSON files (`random_synthetic_profiles_qwen` or `symptoms_synthetic_profiles_qwen`) |

```bash
python script_synth_profile_sessions.py --profile_source_dir random_synthetic_profiles_qwen   -o out_random_synth -model llama
python script_synth_profile_sessions.py --profile_source_dir symptoms_synthetic_profiles_qwen  -o out_symptoms_synth -model llama
```
