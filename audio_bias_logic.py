def compute_raw_score(row, const=-0.0125):
    return (
        const
        + (0.0001 * row.get("count_male", 0))
        + (-0.0001 * row.get("count_female", 0))
        + (0.0005 * row.get("voice_activity_male", 0))
        + (-0.0005 * row.get("voice_activity_female", 0))
        + (0.0044 * row.get("energy_male", 0))
        + (-0.0034 * row.get("energy_female", 0))
        + (-0.0004 * row.get("amplitude_male", 0))
        + (-0.0004 * row.get("amplitude_female", 0))
        + (-0.0334 * row.get("pitch_male", 0))
        + (-0.0325 * row.get("pitch_female", 0))
        + (0.0002 * row.get("pitch_male", 0) ** 2)
        + (-0.0002 * row.get("pitch_male", 0) * row.get("pitch_female", 0))
        + (0.0002 * row.get("pitch_female", 0) ** 2)
    )

def min_max_scale(value, min_value, max_value, scale_min=0, scale_max=10, preserve_sign=False):
    if max_value == min_value:
        return scale_min
    scaled_value = scale_min + ((value - min_value) / (max_value - min_value)) * (scale_max - scale_min)
    return scaled_value if not preserve_sign else (-scaled_value if value < 0 else scaled_value)

def calculate_score_from_form(row, const=-0.0125):
    raw_score = compute_raw_score(row, const)

    if raw_score < 0:
        female_max = row.copy()
        female_min = row.copy()

        for key in row:
            if key.endswith("_male"):
                female_max[key] = 0
                female_min[key] = row.get(key.replace("_male", "_female"), 0)

        max_score = compute_raw_score(female_max, const)
        min_score = compute_raw_score(female_min, const)
        scaled = min_max_scale(raw_score, min_score, max_score, preserve_sign=True)

    else:
        male_max = row.copy()
        male_min = row.copy()

        for key in row:
            if key.endswith("_female"):
                male_max[key] = 0
                male_min[key] = row.get(key.replace("_female", "_male"), 0)

        max_score = compute_raw_score(male_max, const)
        min_score = compute_raw_score(male_min, const)
        scaled = min_max_scale(raw_score, min_score, max_score, preserve_sign=True)

    # Set bias message based on scaled score
    if abs(scaled) < 0.00001:
        bias_message = "The dataset appears gender-neutral (scaled score ≈ 0)."
    elif scaled > 0:
        bias_message = f"The dataset is biased toward male by {round(abs(scaled), 5)} (scaled score)."
    else:
        bias_message = f"The dataset is biased toward female by {round(abs(scaled), 5)} (scaled score)."

    return {
        "raw_score": round(raw_score, 5),
        "scaled_score": round(scaled, 5),
        "min_score": round(min_score, 5),
        "max_score": round(max_score, 5),
        "bias_message": bias_message
    }

