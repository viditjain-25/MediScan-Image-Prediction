import os
import sqlite3
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =====================================
# 1️⃣ DATABASE LOADING (ONCE)
# =====================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "mediscan.db")

conn = sqlite3.connect(DB_PATH)

df = pd.read_sql("SELECT * FROM diseases", conn)
description_data = pd.read_sql("SELECT * FROM descriptions", conn)
precaution_data = pd.read_sql("SELECT * FROM precautions", conn)
severity_data = pd.read_sql("SELECT * FROM symptoms", conn)

conn.close()


# =====================================
# 2️⃣ NORMALIZATION
# =====================================
df["Disease"] = df["Disease"].str.lower()
description_data["Disease"] = description_data["Disease"].str.lower()
precaution_data["Disease"] = precaution_data["Disease"].str.lower()
severity_data["Symptom"] = severity_data["Symptom"].str.lower()

symptom_cols = [c for c in df.columns if c.lower().startswith("symptom")]
precaution_cols = [c for c in precaution_data.columns if c.lower().startswith("precaution")]


# =====================================
# 3️⃣ PRECOMPUTATION (SPEED BOOST)
# =====================================

# Combine symptoms per disease
df["all_symptoms"] = (
    df[symptom_cols]
    .fillna("")
    .apply(lambda x: " ".join(x), axis=1)
    .str.lower()
)

# Precompute known symptom SET per disease (🔥 huge speed boost)
df["known_symptoms"] = df[symptom_cols].apply(
    lambda x: set(
        s.strip().lower()
        for s in x.dropna()
    ),
    axis=1
)

# Combine precautions text
precaution_data["all_Precaution"] = (
    precaution_data[precaution_cols]
    .fillna("")
    .apply(lambda x: " ".join(x), axis=1)
    .str.lower()
)


# =====================================
# 4️⃣ TF-IDF MODEL (CACHED)
# =====================================
# char + word ngrams → handles typos + semantics
vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 2),
    min_df=1
)

tfidf_matrix = vectorizer.fit_transform(df["all_symptoms"])


# =====================================
# 5️⃣ SYMPTOM ALIASES
# =====================================
SYMPTOM_ALIASES = {
    "fever": ["high_fever", "mild_fever"],  
    "vomiting": ["vomiting", "nausea"],
    "cold": ["runny_nose", "chills", "congestion"],
    "stomach_ache": ["abdominal_pain", "stomach_pain"],
    "headache": ["headache"],
    "pain": ["muscle_pain", "joint_pain", "back_pain"]
}


def expand_symptoms(symptoms):
    expanded = []
    for s in symptoms:
        expanded.extend(SYMPTOM_ALIASES.get(s, [s]))
    return list(set(expanded))


# =====================================
# 6️⃣ MAIN PREDICTION FUNCTION
# =====================================
def predict_disease(user_input, top_n=4):
    if not user_input:
        return []

    # ---------------------
    # Clean input
    # ---------------------
    user_symptoms = [
        s.strip().lower().replace(" ", "_")
        for s in user_input.split(",")
        if s.strip()
    ]

    expanded = expand_symptoms(user_symptoms)
    if not expanded:
        return []

    # ---------------------
    # Vector similarity
    # ---------------------
    user_text = " ".join(expanded)
    user_vec = vectorizer.transform([user_text])
    similarity = cosine_similarity(user_vec, tfidf_matrix).flatten()

    # Top-K pruning (speed)
    TOP_K = 10
    top_indices = similarity.argsort()[-TOP_K:][::-1]

    disease_scores = {}

    for idx in top_indices:
        sim_score = similarity[idx]
        if sim_score == 0:
            continue

        row = df.iloc[idx]
        disease = row["Disease"]
        known = row["known_symptoms"]

        matches = set(sym for sym in expanded if sym in known)
        non_matches = set(expanded) - matches

        if disease not in disease_scores:
            disease_scores[disease] = {
                "match_count": 0,
                "matching": set(),
                "non_matching": set(expanded),
                "similarity": 0
            }

        disease_scores[disease]["match_count"] += len(matches)
        disease_scores[disease]["matching"].update(matches)
        disease_scores[disease]["non_matching"].difference_update(matches)
        disease_scores[disease]["similarity"] = max(
            disease_scores[disease]["similarity"],
            sim_score
        )

    # ---------------------
    # Build results
    # ---------------------
    results = []

    for disease, info in disease_scores.items():

        desc_row = description_data[description_data["Disease"] == disease]
        precaution_row = precaution_data[precaution_data["Disease"] == disease]
        severity_match = severity_data[
            severity_data["Symptom"].isin(info["matching"])
        ]

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

        # Severity calculation
        if not severity_match.empty:
            avg = severity_match["Severity"].astype(int).mean()
            severity = "Severe" if avg >= 6 else "Moderate" if avg >= 3 else "Mild"
        else:
            severity = "Unknown"

        # 🎯 Improved probability (accuracy boost)
        probability = round(
            (info["match_count"] / max(len(expanded), 1)) * 100,
            2
        )

        # ❌ penalty for non-matching symptoms
        penalty = len(info["non_matching"]) * 0.5
        final_score = (info["match_count"] + info["similarity"] * 2) - penalty

        results.append({
            "disease": disease.title(),
            "description": description,
            "precaution": precaution,
            "severity": severity,
            "matching": list(info["matching"]),
            "non_matching": list(info["non_matching"]),
            "probability": probability,
            "score": round(final_score, 2)
        })

    # ---------------------
    # Fallback
    # ---------------------
    if not results:
        best_idx = similarity.argmax()
        fallback = df.iloc[best_idx]["Disease"]

        desc = description_data[description_data["Disease"] == fallback]["Description"]
        precaution = precaution_data[precaution_data["Disease"] == fallback]["all_Precaution"]

        return [{
            "disease": fallback.title(),
            "description": desc.values[0] if not desc.empty else "No description available.",
            "precaution": precaution.values[0] if not precaution.empty else "No precaution available.",
            "severity": "Unknown",
            "matching": [],
            "non_matching": expanded,
            "probability": 0.0,
            "score": 0
        }]

    # Final sort & limit
    return sorted(results, key=lambda x: -x["score"])[:top_n]