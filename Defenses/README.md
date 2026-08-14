# Defenses

Scripts that build *defended* (anonymized/perturbed) versions of the Eeyore profile data, and a post-hoc defense applied directly to already-generated sessions. All three algorithm scripts below hardcode their input paths to `eeyore-data.parquet`/`valid_indices.json` at the repository root (resolved from the script's own location, not the current directory), so they can be run from anywhere and take no `--input`/`--valid-idx` flags; only the `-o/--output` path is resolved relative to the current directory.

## Defense Algorithms

Each of these three scripts reads `eeyore-data.parquet` and writes a new, defended profile parquet file that [`Profile/generate_profile_sessions.py`](Profile/README.md) and [`Situation/generate_situation_sessions.py`](Situation/README.md) then generate sessions from.

| Defense | Script | Output |
|---|---|---|
| Coarsened Profile (CoarsProf) | `CoarsProf.py` | `eeyore-data-anon.parquet` |
| Noisy Profile (NoiseProf) | `NoiseProf.py` | `eeyore-dp-simple.parquet` |
| Noisy Profile Iterative Mixing (NoiseIterMix) | `NoiseIterMix.py` | `eeyore-dp-mix-iter.parquet` |

```bash
python Defenses/CoarsProf.py \
    -o eeyore-data-anon.parquet -model qwen

python Defenses/NoiseProf.py \
    -o eeyore-dp-simple.parquet -m qwen -p 0.3

python Defenses/NoiseIterMix.py \
    -o eeyore-dp-mix-iter.parquet -m qwen -p 0.2
```

`CoarsProf.py` takes `-o/--output` (default `eeyore-data-anon.parquet`), `-model/--model_name` (default `llama`), `--batch-size` (default `50`; rows per checkpoint — the output parquet is written incrementally and a partial run resumes from where it left off), and `--no-llm` (flag; skips the LLM-based situation rewrite and only applies rule-based occupation/marital-status generalisation and PII field dropping).

NoiseProf and NoiseIterMix take `-m/--model_name` (default `llama`), `-o/--output`, `--seed` (default `42`), and `-p/--noise-prob` (default `0.7`; the probability that each demographic attribute **is** perturbed — the paper's `p ∈ {0.2, 0.5, 0.8}` maps directly onto this flag). NoiseIterMix additionally accepts `-th/--sim-threshold` (default `0.6`) and `--max-iters` (default `5`) controlling how far the situation must diverge from the original, `--st-model` (default `all-MiniLM-L6-v2`, the sentence-transformers model used to measure that divergence), and `--no-references` (flag; disables cross-referencing other profiles' situations when mixing). `--len-tol` (default `1.15`) caps how much longer the mixed situation may be relative to the original.

## NER-based De-identification (post-hoc)

Unlike the three algorithms above, `NER.py` runs on already-*generated* session JSON files (FP or CS), not on the seed profile parquet. It always filters input files against `valid_indices.json` at the repository root — a `session_<i>.json` file is only processed if `(i - 1)` is a valid index; this is unconditional, not a flag.

| Argument | Description |
|---|---|
| `input` | Session JSON file(s) or glob, e.g. `'sessions/*.json'` |
| `-m` / `--model_name` | Model registered in vLLM (default `llama`) |
| `-o` / `--output-dir` | Where to write anonymized sessions (default `./anonymized`) |
| `-d` / `--delay` | Seconds to sleep between requests (default `0.0`) |

The vLLM server URL is hardcoded (`common/llm_client.VLLM_BASE_URL`, `http://localhost:8000/v1`) — there is no `--base-url`/`--api-key` override.

```bash
# FP sessions
python Defenses/NER.py 'out_fp/*.json' -o out_fp_ner -m qwen

# CS sessions
python Defenses/NER.py 'out_cs/*.json' -o out_cs_ner -m qwen
```

## Generating Defended Sessions

Once a defended parquet exists, generate sessions from it with [`Profile/generate_profile_sessions.py`](Profile/README.md) (for FP) or [`Situation/generate_situation_sessions.py`](Situation/README.md) (for CS).
