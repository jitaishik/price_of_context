# Session Generation Prompts

Prompt templates for generating synthetic counseling sessions at each of the three privacy levels (see the [root README](../README.md)). Each file is a [LangChain `PromptTemplate`](https://python.langchain.com/) with the input variables listed below, built from a patient's Eeyore profile by [`common/profile_attributes.get_client_information`](../common/profile_attributes.py).

| Prompt file | Privacy level | Used by | Input variables |
|---|---|---|---|
| `symptoms.txt` | Symptom-Only (SO) | `Generation/script_symptoms.py` | `client_medical_information`, `client_resistance` |
| `situation.txt` | Contextual Situation (CS) | `Generation/script_situation.py`, `Defenses/Situation/generate_situation_sessions.py` | `client_background_information`, `client_medical_information`, `client_resistance` |
| `profile.txt` | Full Profile (FP) | `Generation/script_profile.py`, `Generation/script_synth_profile_sessions.py`, `Defenses/Profile/generate_profile_sessions.py` | `client_background_information`, `client_medical_information`, `client_resistance` |

`client_background_information` contains demographics + situation for `profile.txt`, but only the situation for `situation.txt` (`get_client_information(profile, mode="situation")` deliberately excludes demographics so CS generation never sees them directly).

See [`Generation/README.md`](../Generation/README.md) for how these prompts are invoked.
