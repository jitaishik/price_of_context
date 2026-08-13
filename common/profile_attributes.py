SYMPTOM_FIELDS = [
    "symptom severity",
    "cognition distortion exhibition",
    "depression severity",
    "suicidal ideation severity",
    "homicidal ideation severity",
]

# Fields that never belong in "background information", regardless of mode.
_NON_BACKGROUND_FIELDS = {
    "resistance toward the support",
    "counseling history",
    *SYMPTOM_FIELDS,
}


def get_client_information(profile, mode="default"):
    """Return (client_background_information, client_medical_information,
    client_resistance, counseling_history) extracted from `profile`.

    `mode="default"`: every field not in _NON_BACKGROUND_FIELDS is treated as
    background information (used for the profile.txt/symptoms.txt prompts,
    i.e. Full Profile and Symptom-Only generation).

    `mode="situation"`: only "situation of the client" is treated as
    background information (used for the situation.txt prompt, i.e.
    Contextual Situation generation, which must not leak demographics).
    """
    client_background_information = ""
    client_medical_information = ""
    client_resistance = "unknown"
    counseling_history = ""

    for attribute in profile.keys():
        if attribute in SYMPTOM_FIELDS:
            value = profile[attribute]
            if isinstance(value, dict):
                client_medical_information += attribute + ": " + "\n"
                for feature in value.keys():
                    client_medical_information += feature + ": " + value[feature] + "\n"
            else:
                client_medical_information += attribute + ": " + value + "\n"
        elif attribute == "resistance toward the support":
            client_resistance = profile[attribute]
        elif attribute == "counseling history":
            counseling_history = profile[attribute]
        elif mode == "situation":
            if attribute == "situation of the client":
                client_background_information += attribute + ": " + profile[attribute] + "\n"
        else:
            client_background_information += attribute + ": " + profile[attribute] + "\n"

    return (
        client_background_information.strip(),
        client_medical_information.strip(),
        client_resistance,
        counseling_history,
    )
