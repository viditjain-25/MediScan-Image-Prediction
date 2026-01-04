import pandas as pd



# -----------------------------
# Load CSV
# -----------------------------
df = pd.read_csv("eye_disease\eye_symptoms.csv")

# Create disease -> symptoms dictionary
disease_symptoms = {}

for _, row in df.iterrows():
    disease = row["Disease"]
    symptoms = set()

    for col in df.columns[1:]:
        if pd.notna(row[col]):
            symptoms.add(row[col].strip().lower())

    disease_symptoms[disease] = symptoms


# -----------------------------
# Prediction Function ONLY
# -----------------------------
def predict_disease_from_symptoms(user_symptoms):
    user_symptoms = {s.strip().lower() for s in user_symptoms if s.strip()}

    if not user_symptoms:
        return None, 0, {}

    scores = {}

    for disease, symptoms in disease_symptoms.items():
        matched = user_symptoms & symptoms

        if matched:
            percentage = (len(matched) / len(user_symptoms)) * 100
            scores[disease] = round(percentage, 2)

    if not scores:
        return None, 0, {}

    best_disease = max(scores, key=scores.get)

    return best_disease, scores[best_disease], scores
