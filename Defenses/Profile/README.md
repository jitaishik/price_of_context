# Defenses / Profile

Generates Full Profile (FP) sessions from a *defended* profile parquet produced by one of the [Defense Algorithms](../README.md#defense-algorithms), instead of the raw `eeyore-data.parquet` used by [`Generation/script_profile.py`](../../Generation/README.md). Otherwise identical generation logic (same `profile.txt` prompt, same `get_client_information(mode="default")` split).

| Argument | Description |
|---|---|
| `-o` / `--output_dir` | Directory to save the generated `session_<i>.json` files |
| `-model` / `--model_name` | Judge/generator model: `llama` or `qwen` |
| `-i` / `--input_file` | **Required.** Path to the defended profile parquet file |

```bash
python generate_profile_sessions.py -i ../eeyore-data-anon.parquet   -o out_fp_coarsprof   -model llama
python generate_profile_sessions.py -i ../eeyore-dp-simple.parquet   -o out_fp_noiseprof    -model llama
python generate_profile_sessions.py -i ../eeyore-dp-mix-iter.parquet -o out_fp_noiseitermix -model llama
```
