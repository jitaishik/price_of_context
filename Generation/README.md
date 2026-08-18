# Generation

Scripts for generating synthetic counseling sessions from the Eeyore seed profiles (`../eeyore-data.parquet`) at each of the three privacy levels, plus scripts for generating and using *synthetic* (non-real) patient profiles as an alternative to real ones.

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

Builds synthetic (non-real) patient profiles as JSON files, one per source Eeyore symptom set. Run from this directory. Both scripts take the same `-o/--output_dir` (default `.`) and `-model/--model_name` (default `llama`) flags as the real-profile scripts above, and each creates its own output directory named `<output_dir>_<model_name>` (e.g. `-o random_synthetic_profiles -model qwen` writes to `random_synthetic_profiles_qwen`). Both load their generation prompts from `profile_gen_prompts/` (`random_system.txt`/`random_user.txt` and `symptoms_system.txt`/`symptoms_user.txt`).

| Variant | Script | Strategy |
|---|---|---|
| Random attributes | `random_synthetic_profiles.py` | Randomly samples age/occupation/marital status/gender, then asks the LLM to generate only a name + situation |
| Symptoms-grounded | `symptoms_synthetic_profiles.py` | Asks the LLM to generate all demographic attributes and a situation from symptoms alone |

```bash
python random_synthetic_profiles.py -o random_synthetic_profiles -model qwen
python symptoms_synthetic_profiles.py -o symptoms_synthetic_profiles -model qwen
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

## Generation Prompts

`profile_gen_prompts/` holds the system/user prompt pairs used above: `random_system.txt` / `random_user.txt` (random-attribute variant) and `symptoms_system.txt` / `symptoms_user.txt` (symptoms-grounded variant).
