def final_decision(image_pred, image_conf, symptom_pred, symptom_conf):
    # Case 1: Image says NORMAL
    if image_pred == "Normal_Eyes" and image_conf >= 75:
        if symptom_conf > 59:
            return {
                "status": "warning",
                "message": "Symptoms suggest an issue, but image appears normal. Medical consultation recommended."
            }
        else:
            return {
                "status": "normal",
                "message": "Eye appears normal."
            }

    # Case 2: Image disease == Symptom disease
    if image_pred == symptom_pred:
        combined_conf = round((image_conf + symptom_conf) / 2, 2)
        return {
            "status": "confirmed",
            "disease": image_pred,
            "confidence": combined_conf,
            "message": "Image and symptoms match."
        }

    # Case 3: Image disease ≠ Symptom disease
    return {
        "status": "conflict",
        "image_prediction": {
            "disease": image_pred,
            "confidence": image_conf
        },
        "symptom_prediction": {
            "disease": symptom_pred,
            "confidence": symptom_conf
        },
        "message": "Image and symptoms do not match. Further examination required."
    }
# -----------------------------
# TEST BLOCK
# -----------------------------
if __name__ == "__main__":
    tests = [
        ("Normal_Eyes", 90, "Cataracts", 70),
        ("Cataracts", 88, "Cataracts", 75),
        ("Cataracts", 88, "Glaucoma", 65)
    ]

    for t in tests:
        print("\nINPUT:", t)
        print("OUTPUT:", final_decision(*t))
