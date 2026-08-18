# Defenses / Situation

Generates Contextual Situation (CS) sessions from a *defended* profile parquet produced by one of the [Defense Algorithms](../README.md#defense-algorithms), instead of the raw `eeyore-data.parquet` used by [`Generation/script_situation.py`](../../Generation/README.md). Otherwise same generation logic (same `situation.txt` prompt, same `get_client_information(mode="situation")` split, which excludes demographics even though they may be present in the defended profile, and same `valid_indices.json` filtering) as `Generation/script_situation.py`.

| Argument | Description |
|---|---|
| `-o` / `--output_dir` | Directory to save the generated `session_<i>.json` files |
| `-model` / `--model_name` | Judge/generator model: `llama` or `qwen` |
| `-i` / `--input_file` | **Required.** Path to the defended profile parquet file |

```bash
python generate_situation_sessions.py -i ../eeyore-data-anon.parquet   -o out_cs_coarsprof   -model llama
python generate_situation_sessions.py -i ../eeyore-dp-simple.parquet   -o out_cs_noiseprof    -model llama
python generate_situation_sessions.py -i ../eeyore-dp-mix-iter.parquet -o out_cs_noiseitermix -model llama
```
