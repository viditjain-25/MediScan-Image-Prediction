import os
import sqlite3
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "mediscan.db")

conn = sqlite3.connect(DB_PATH)
df = pd.read_sql("SELECT * FROM diseases", conn)



description_data = pd.read_sql("SELECT * FROM descriptions", conn)
precaution_data = pd.read_sql("SELECT * FROM precautions", conn)
severity_data = pd.read_sql("SELECT * FROM symptoms", conn)

# Normalize strings
df["Disease"] = df["Disease"].str.lower()
description_data["Disease"] = description_data["Disease"].str.lower()
precaution_data["Disease"] = precaution_data["Disease"].str.lower()
severity_data["Symptom"] = severity_data["Symptom"].str.lower()

# ---------------------
# 2. Preprocess
# ---------------------
symptom_cols = [col for col in df.columns if col.lower().startswith("symptom")]
precaution_cols = [col for col in precaution_data.columns if col.lower().startswith("precaution")]

df["all_symptoms"] = (
    df[symptom_cols]
    .fillna("")
    .apply(lambda x: " ".join(x), axis=1)
    .str.lower()
)

precaution_data["all_Precaution"] = (
    precaution_data[precaution_cols]
    .fillna("")
    .apply(lambda x: " ".join(x), axis=1)
    .str.lower()
)

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["all_symptoms"])

# ---------------------
# 3. Symptom Aliases
# ---------------------
SYMPTOM_ALIASES = {
    "fever": ["high_fever", "mild_fever"],
    "vomiting": ["vomiting", "nausea"],
    "cold": ["runny_nose", "chills", "congestion"],
    "stomach ache": ["abdominal_pain", "stomach_pain"],
    "headache": ["headache"],
    "pain": ["muscle_pain", "joint_pain", "back_pain"]
}

def expand_symptoms(symptoms):
    expanded = []
    for s in symptoms:
        if s in SYMPTOM_ALIASES:
            expanded.extend(SYMPTOM_ALIASES[s])
        else:
            expanded.append(s)
    return list(set(expanded))

# ---------------------
# 4. Predict Function
# ---------------------
def predict_disease(user_input, top_n=4):
    if not user_input:
        return []

    # Clean user input
    user_symptoms = [
        s.strip().lower().replace(" ", "_")
        for s in user_input.split(",")
        if s.strip()
    ]

    expanded = expand_symptoms(user_symptoms)
    if not expanded:
        return []

    user_text = " ".join(expanded)
    user_vec = vectorizer.transform([user_text])
    similarity = cosine_similarity(user_vec, tfidf_matrix).flatten()

    disease_scores = {}

    for idx, score in enumerate(similarity):
        if score == 0:
            continue

        row = df.iloc[idx]
        disease = row["Disease"]

        known = [
            str(row[col]).strip().lower()
            for col in symptom_cols
            if pd.notna(row[col])
        ]

        matches = set(sym for sym in expanded if sym in known)

        if disease not in disease_scores:
            disease_scores[disease] = {
                "score": 0,
                "matching": set(),
                "non_matching": set(expanded),
                "rows": []
            }

        disease_scores[disease]["score"] += len(matches)
        disease_scores[disease]["matching"].update(matches)

        # ✅ Non-matching symptoms logic
        disease_scores[disease]["non_matching"].difference_update(matches)

        disease_scores[disease]["rows"].append(idx)

    results = []
    for disease, info in disease_scores.items():
        desc_row = description_data[description_data["Disease"] == disease]
        precaution_row = precaution_data[precaution_data["Disease"] == disease]
        severity_match = severity_data[severity_data["Symptom"].isin(info["matching"])]

        description = (
            desc_row["Description"].values[0]
            if not desc_row.empty
            else "No description available."
        )

        precaution = (
            precaution_row["all_Precaution"].values[0]
            if not precaution_row.empty
            else "No precautions available."
        )

        if not severity_match.empty:
            avg = severity_match["Severity"].astype(int).mean()
            severity = "Severe" if avg >= 6 else "Moderate" if avg >= 3 else "Mild"
        else:
            severity = "Unknown"

        results.append({
            "disease": disease.title(),
            "description": description,
            "precaution": precaution,
            "severity": severity,
            "matching": list(info["matching"]),
            "non_matching": list(info["non_matching"]),
            "probability": round((len(info["matching"]) / len(expanded)) * 100, 2),
            "score": info["score"]
        })

    # ---------------------
    # 5. Fallback Case
    # ---------------------
    if not results:
        best_idx = similarity.argmax()
        fallback_disease = df.iloc[best_idx]["Disease"]

        desc = description_data[description_data["Disease"] == fallback_disease]["Description"]
        precaution = precaution_data[precaution_data["Disease"] == fallback_disease]["all_Precaution"]

        return [{
            "disease": fallback_disease.title(),
            "description": desc.values[0] if not desc.empty else "No description available.",
            "precaution": precaution.values[0] if not precaution.empty else "No precaution available.",
            "severity": "Unknown",
            "matching": [],
            "non_matching": expanded,
            "probability": 0.0,
            "score": 0
        }]

    return sorted(results, key=lambda x: (-x["score"]))[:top_n]
