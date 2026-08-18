# Evaluation

Utility (quality, diversity) and privacy (attribute extraction, membership inference, linkage) evaluation of generated synthetic counseling sessions.

## Scripts

| Script | Purpose | Key arguments |
|---|---|---|
| `attribute_extraction.py` | Attribute extraction attack: recovers name/gender/age/occupation/marital status from a session via LLM-as-judge; gender/age/occupation/marital status are each chosen from a fixed label set, name is extracted open-ended (verbatim or `'Cannot be identified'`) | `-i/--input_dir`, `-o/--output_dir`, `-model/--model_name` (`llama`\|`qwen`\|`gpt`) |
| `situation_extraction.py` | Extracts the patient's pre-help-seeking situation from a session, for situation-leakage / linkage / MIA measurements | `-i/--input_dir`, `-o/--output_file`, `-o_dir/--output_dir` (default `.`), `-model/--model_name` (`llama`\|`qwen`\|`gpt`, default `llama`) |

```bash
python attribute_extraction.py -i ../Generation/out_fp -o attr_out -model qwen
python situation_extraction.py -i ../Generation/out_fp -o situations.json -o_dir extracted_situations -model qwen
```

Counseling-quality scoring (CTRS, WAI) lives in their own subfolders: see [`CTRS/README.md`](CTRS/README.md) and [`WAI/README.md`](WAI/README.md).

## Notebooks

| Notebook | Purpose |
|---|---|
| `diversity.ipynb` | Distinct-n, Expected Adjusted Distinct (EAD), Entropy-n, Lexical Diversity Density (LDD), and average pairwise cosine similarity between generated patient situations (SitSim) over generated sessions |
| `sim_2_real_sit.ipynb` | Cosine similarity between each session's extracted situation and its ground-truth (real) situation — measures how well `situation_extraction.py` recovers the real situation, not diversity among generated situations |
| `attribute_extraction_results.ipynb` | Scores `attribute_extraction.py` output against ground-truth profiles: raw TP/FP/TN/FN counts and an attribute leakage rate, overall and per-attribute |
| `MIA.ipynb` | Membership inference attack: AUC-ROC and TPR @ low-FPR thresholds using `attribute_extraction.py`/`situation_extraction.py` output as the reference set |
| `Linkage.ipynb` | Linkage attack: matches extracted session attributes/situations back to source patient profiles |

`situation_non_member_valid_dedup_2_th_100.json` is the held-out non-member situation set used by `MIA.ipynb`.
