# Defenses

Scripts that build *defended* (anonymized/perturbed) versions of the Eeyore profile data, and a post-hoc defense applied directly to already-generated sessions. Run the algorithm scripts below from the repository root (they default to reading `eeyore-data.parquet` relative to the current directory).

## Defense Algorithms

Each of these three scripts reads `eeyore-data.parquet` and writes a new, defended profile parquet file that [`Profile/generate_profile_sessions.py`](Profile/README.md) and [`Situation/generate_situation_sessions.py`](Situation/README.md) then generate sessions from.

| Defense | Script | Output |
|---|---|---|
| Coarsened Profile (CoarsProf) | `generalize_profiles.py` | `eeyore-data-anon.parquet` |
| Noisy Profile (NoiseProf) | `create_eeyore_dp_simple.py` | `eeyore-dp-simple.parquet` |
| Noisy Profile Iterative Mixing (NoiseIterMix) | `create_eeyore_dp_mix_iter.py` | `eeyore-dp-mix-iter.parquet` |

```bash
python Defenses/generalize_profiles.py \
    --input eeyore-data.parquet --output eeyore-data-anon.parquet \
    --model qwen

python Defenses/create_eeyore_dp_simple.py \
    --input eeyore-data.parquet --valid-idx valid_indices.json \
    --output eeyore-dp-simple.parquet --model path_to_model --keep-prob 0.7

python Defenses/create_eeyore_dp_mix_iter.py \
    --input eeyore-data.parquet --valid-idx valid_indices.json \
    --output eeyore-dp-mix-iter.parquet --model path_to_model --keep-prob 0.8
```

NoiseProf and NoiseIterMix accept `--keep-prob` (probability of leaving each demographic attribute unchanged; the paper evaluates `p ∈ {0.2, 0.5, 0.8}`, reported as perturbation probability `1 - keep-prob`). NoiseIterMix additionally accepts `--sim-threshold` (default `0.6`) and `--max-iters` (default `5`) controlling how far the situation must diverge from the original.

## NER-based De-identification (post-hoc)

Unlike the three algorithms above, `anonymize_sessions_ner.py` runs on already-*generated* session JSON files (FP or CS), not on the seed profile parquet.

| Argument | Description |
|---|---|
| `input` | Session JSON file(s) or glob, e.g. `'sessions/*.json'` |
| `--model` / `-m` | Model registered in vLLM |
| `--output-dir` / `-o` | Where to write anonymized sessions (default `./anonymized`) |
| `--valid-indices` / `-V` | Optional: JSON file of valid 0-based indices; if given, only sessions whose `(i - 1)` is listed are processed |

```bash
# FP sessions: process every file
python Defenses/anonymize_sessions_ner.py 'out_fp/*.json' -o out_fp_ner -m qwen_model_path

# CS sessions: restrict to the evaluation subset
python Defenses/anonymize_sessions_ner.py 'out_cs/*.json' -o out_cs_ner -m qwen_model_path --valid-indices valid_indices.json
```

## Generating Defended Sessions

Once a defended parquet exists, generate sessions from it with [`Profile/generate_profile_sessions.py`](Profile/README.md) (for FP) or [`Situation/generate_situation_sessions.py`](Situation/README.md) (for CS).
