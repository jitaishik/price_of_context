# The Price of Context: Privacy-Utility Trade-offs in Synthetic Counseling Session Generation

## Summary

This codebase provides the scripts used in the experiments of the paper **"The Price of Context: Privacy-Utility Trade-offs in Synthetic Counseling Session Generation"** (currently under anonymous review at ACL). General-purpose LLMs underperform on counseling tasks, motivating specialized models trained on sensitive real-world counseling data that is rarely available due to privacy constraints. Synthetic counseling session generation offers a way around this, but the privacy-utility trade-off across different choices of private input remains underexplored. We systematically study this trade-off by generating synthetic counseling sessions from patient profiles in the [Eeyore](https://huggingface.co/datasets/liusiyang/eeyore_profile) dataset at three privacy levels:

* **Symptom-Only (SO):** Only a list of symptoms and their severity levels — no demographic or situational context. The highest-privacy regime.
* **Contextual Situation (CS):** Symptoms plus the patient's background situational description. No explicit quasi-identifiers, but situations can still leak information indirectly.
* **Full Profile (FP):** Symptoms, situation, and demographics (age, gender, occupation, marital status). The lowest-privacy, richest-context regime.

We evaluate **utility** through counseling quality (LLM-as-a-judge on CTRS and WAI) and diversity (Distinct-n, EAD, Entropy-n, LDD, and situation cosine similarity), and **privacy** by adopting a data-holder perspective and measuring disclosure risk through three LLM-based attacks:

* **Attribute Extraction Attack:** Prompts an LLM to recover demographic attributes and situations from generated sessions.
* **Membership Inference Attack (MIA):** Determines whether a given patient profile was used to generate a session, via similarity to a reference set of extracted profiles.
* **Linkage Attack:** Matches a synthetic session back to its source patient profile among all candidate profiles.

We find that richer private context (situations, demographics) **improves diversity but does not improve counseling quality**, while **substantially increasing privacy risk**. We further evaluate four defenses:

* **NER-based De-identification (NER):** Post-hoc LLM-based NER pipeline that replaces PII (names, age, occupation, marital status, organizations, gender) with categorical placeholders in generated sessions.
* **Coarsened Profile (CoarsProf):** Pre-generation defense that coarsens profile attributes (drops names, buckets occupation/marital status, keeps age/gender categorical) and adapts the situation to match via an LLM.
* **Noisy Profile (NoiseProf):** Pre-generation defense — proposed in this work — that independently perturbs each profile attribute with probability *p* by resampling from the attribute pool, then adapts the situation to stay coherent.
* **Noisy Profile Iterative Mixing (NoiseIterMix):** Pre-generation defense — proposed in this work — that builds on NoiseProf by iteratively mixing in situation content from other profiles sharing a demographic attribute until divergence from the original situation (cosine similarity < 0.6, up to 5 iterations).

> **Abstract:** The rise in mental health disorders has increased interest in LLM-based counseling agents. However, general-purpose LLMs underperform on counseling tasks, motivating the use of specialized models trained on sensitive real-world counseling data, which is often unavailable due to privacy constraints. Synthetic counseling session generation offers a potential solution, but the privacy–utility trade-off involved in different private input choices remains underexplored. In this work, we systematically study this trade-off across three privacy levels: SO (symptoms only), CS (symptoms + situation), and FP (symptoms + situation + demographics). We evaluate utility through psychological quality and diversity, and privacy through attribute extraction, membership inference, and linkage attacks. Results show that incorporating situations improves diversity, but substantially increases privacy risks. We further evaluate defenses, including NER-based redaction, demographic coarsening and situation generalization, and two proposed noise-based methods: demographic perturbation and iterative situation reframing. While simple defenses provide limited protection over FP generations, our noise-based methods reduce linkage and membership inference attack effectiveness by 20% each and attribute extraction leakage by 35%, while preserving diversity.

## Key Findings

* **Privacy level vs. quality:** SO, CS, and FP achieve comparable CTRS/WAI scores — reducing patient information has limited impact on counseling quality.
* **Privacy level vs. diversity:** CS and FP consistently improve lexical and situational diversity over SO; situational information (not demographics) is the main driver.
* **Privacy level vs. leakage:** FP allows recovery of ~59% of demographic attributes and achieves >0.95 AUC on linkage; CS leaks ~47% of attributes even without explicit demographics, showing situations implicitly reveal demographic information.
* **Defenses:** Surface-level defenses (NER, CoarsProf) offer only limited protection due to indirect leakage. The proposed noise-based defenses (NoiseProf, NoiseIterMix) are more effective — NoiseIterMix reduces MIA and linkage AUC and attribute leakage substantially while preserving diversity comparable to FP.

Contact: see the paper for author/contact details (currently under anonymous review).

## Setting up the environment

Session generation, evaluation, and defenses in this repo call a **local vLLM OpenAI-compatible server** (`http://localhost:8000/v1`) hosting the generator/judge model, so start your vLLM server first, e.g.:

```bash
python -m vllm.entrypoints.openai.api_server --model <path_to_model> --port 8000
```

Then set up the Python environment (no `requirements.txt` is provided; install the libraries used across the codebase, listed in [References/Libraries](#referenceslibraries), plus `pandas`, `numpy`, `scikit-learn`, and `matplotlib`):

```bash
python3 -m venv price_of_context_env
source price_of_context_env/bin/activate
pip install vllm langchain langchain-openai sentence-transformers huggingface_hub pandas numpy scikit-learn matplotlib
```

## Repository Structure

Each folder below has its own README with argument tables and example commands.

```
Price_Of_context/
│
├── eeyore-data.parquet                   # Source patient profiles (Eeyore dataset)
├── valid_indices.json                    # Indices of the 310 valid seed profiles used
├── valid_indices_all.json                # Indices of all candidate profiles considered
├── valid_non_member_indices_dedup_2_th_100.json  # Held-out non-member profile indices (for MIA)
│
├── common/                               # Shared LLM client / prompt / retry / profile-parsing helpers
│   ├── llm_client.py                     #   vLLM/OpenAI client construction
│   ├── prompt_utils.py                   #   load_prompt / save_as_json / fix_encoding_txt
│   ├── json_retry.py                     #   JSON-retry and exception-backoff helpers
│   └── profile_attributes.py             #   Split a raw profile into background/medical/resistance strings
│
├── prompts_script/                       # Core session-generation prompts (see prompts_script/README.md)
│   ├── symptoms.txt                      #   SO prompt (symptoms only)
│   ├── situation.txt                     #   CS prompt (symptoms + situation)
│   └── profile.txt                       #   FP prompt (symptoms + situation + demographics)
│
├── Generation/                           # Synthetic session generation (see Generation/README.md)
│   ├── script_symptoms.py                #   Generate SO sessions
│   ├── script_situation.py               #   Generate CS sessions
│   ├── script_profile.py                 #   Generate FP sessions
│   ├── script_synth_profile_sessions.py  #   FP sessions from synthetic (non-real) profiles (Appendix E)
│   ├── random_synthetic_profiles.py      #   Build random-attribute synthetic profiles
│   ├── symptoms_synthetic_profiles.py    #   Build symptom-grounded synthetic profiles
│   └── profile_gen_prompts/              #   Prompts for synthetic profile generation
│
├── Defenses/                             # Pre-generation and post-hoc defenses (see Defenses/README.md)
│   ├── CoarsProf.py                      #   Build CoarsProf-anonymized profile dataset
│   ├── NoiseProf.py                      #   Build NoiseProf-perturbed profile dataset
│   ├── NoiseIterMix.py                   #   Build NoiseIterMix profile dataset
│   ├── NER.py                            #   Post-hoc NER redaction on generated sessions (FP or CS)
│   ├── Profile/                          #   Generate FP sessions from a defended profile parquet
│   │   └── generate_profile_sessions.py
│   └── Situation/                        #   Generate CS sessions from a defended profile parquet
│       └── generate_situation_sessions.py
│
└── Evaluation/                           # Utility and privacy evaluation (see Evaluation/README.md)
    ├── diversity.ipynb                   #   Distinct-n, EAD, Entropy-n, LDD, SitSim
    ├── sim_2_real_sit.ipynb              #   Situation similarity analysis
    ├── attribute_extraction.py           #   Attribute extraction attack
    ├── attribute_extraction_results.ipynb#   Attribute extraction attack results/metrics
    ├── situation_extraction.py           #   Situation extraction (feeds attribute/linkage attacks)
    ├── situation_non_member_valid_dedup_2_th_100.json  # Non-member situations for MIA
    ├── MIA.ipynb                         #   Membership inference attack
    ├── Linkage.ipynb                     #   Linkage attack
    ├── CTRS/                             #   Cognitive Therapy Rating Scale (LLM-as-judge)
    │   ├── ctrs.py
    │   ├── rating.ipynb
    │   └── prompts/                      #     One prompt per CTRS dimension (General + CBT-specific)
    └── WAI/                              #   Working Alliance Inventory (LLM-as-judge)
        ├── wai.py
        ├── rating.ipynb
        └── prompts/                      #     One prompt per WAI item (Task/Goal/Bond)
```

## Downloading Data

| Dataset | Link |
|---|---|
| **Eeyore** (seed patient profiles) | [HuggingFace](https://huggingface.co/datasets/liusiyang/eeyore_profile) |

`eeyore-data.parquet` in this repo already contains the seed profiles used for the experiments; `valid_indices.json` selects the 310 unique profiles (with complete demographic information, no duplicates) used to generate synthetic sessions, and `valid_non_member_indices_dedup_2_th_100.json` / `situation_non_member_valid_dedup_2_th_100.json` hold the disjoint non-member set used for membership inference. `valid_indices_all.json` lists all candidate profile indices considered before filtering down to `valid_indices.json`; it is kept for reference but not read by any script in this repo.

## Reproducing Experiments

**1. Generate synthetic sessions at each privacy level**

```bash
cd Generation
python script_symptoms.py  -o out_so -model llama    # SO
python script_situation.py -o out_cs -model llama     # CS
python script_profile.py   -o out_fp -model llama     # FP
```

**2. Evaluate counseling quality (CTRS / WAI)**

```bash
cd Evaluation/CTRS && python ctrs.py -i ../../Generation/out_fp -o ctrs_out -model gpt
cd Evaluation/WAI  && python wai.py  -i ../../Generation/out_fp -o wai_out  -model gpt
```

**3. Evaluate diversity**

Run `Evaluation/diversity.ipynb` (Distinct-n, EAD, Entropy-n, LDD) and `Evaluation/sim_2_real_sit.ipynb` (SitSim) against generated sessions.

**4. Run privacy attacks**

```bash
cd Evaluation
python situation_extraction.py -i <sessions_dir> -o extracted_situations.json -model <model_name>
python attribute_extraction.py -i <sessions_dir> -o extracted_attributes
```
Then score membership inference and linkage using `MIA.ipynb` and `Linkage.ipynb`.

**5. Build defended profile datasets**

Each script hardcodes its input to `eeyore-data.parquet`/`valid_indices.json` at the repository root, so it can be run from anywhere with just an output path (`-o`/`--output`, resolved relative to the current directory):

```bash
python Defenses/CoarsProf.py    -o eeyore-data-anon.parquet   -model llama   # CoarsProf
python Defenses/NoiseProf.py    -o eeyore-dp-simple.parquet   -m llama -p 0.3    # NoiseProf
python Defenses/NoiseIterMix.py -o eeyore-dp-mix-iter.parquet -m llama -p 0.2    # NoiseIterMix
```
`-p`/`--noise-prob` is the probability that each demographic attribute is perturbed (the paper's `p ∈ {0.2, 0.5, 0.8}` maps directly onto this flag).

**6. Generate defended sessions and re-run evaluation**

```bash
cd Defenses/Profile
python generate_profile_sessions.py -i ../../eeyore-data-anon.parquet   -o out_fp_coarsprof   -m llama
python generate_profile_sessions.py -i ../../eeyore-dp-simple.parquet   -o out_fp_noiseprof   -m llama
python generate_profile_sessions.py -i ../../eeyore-dp-mix-iter.parquet -o out_fp_noiseitermix -m llama

cd ../..
python Defenses/NER.py 'Defenses/Profile/out_fp_coarsprof/*.json' -o out_fp_ner -m qwen
```
Repeat evaluation steps 2-4 on each defended output directory. The same pattern applies under `Defenses/Situation/generate_situation_sessions.py` for defenses on CS generations (Appendix G); `NER.py` always filters processed sessions against `valid_indices.json` at the repository root.

## Cite

This paper is currently under anonymous review; a citation will be added once the paper is de-anonymized/published.

## References/Libraries

- [vLLM](https://docs.vllm.ai/en/latest/)
- [LangChain](https://www.langchain.com/)
- [Sentence-Transformers](https://www.sbert.net/)
- [HuggingFace](https://huggingface.co/)
- [Eeyore](https://huggingface.co/datasets/liusiyang/eeyore_profile)

## Disclaimer

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication.
