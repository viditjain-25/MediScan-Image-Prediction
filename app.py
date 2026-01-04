from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from symptoms import predict_disease
import os
import tempfile

from eye_disease.image_predicts import predict_image_from_bytes
from eye_disease.symptoms_predicts import predict_disease_from_symptoms
from eye_disease.decision_engine import final_decision
from eye_disease.eye_validator import is_valid_eye_image

app = Flask(__name__)
CORS(app)


# Load model once (IMPORTANT)
model = tf.keras.models.load_model("model.h5")


@app.route("/", methods=["GET"])
def home():
    return "MediScan ML API is running."


# ---------------------------------
# 1️⃣ GENERAL SYMPTOM DISEASE MODEL
# ---------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        symptoms = data.get("symptoms", "")
        print("Received symptoms:", symptoms)

        result = predict_disease(symptoms)
        return jsonify(result), 200

    except Exception as e:
        print("Exception in /predict:", e)
        return jsonify({"error": str(e)}), 500


# ---------------------------------
# 2️⃣ EYE DISEASE MODEL (IMAGE / SYMPTOMS / BOTH)
# ---------------------------------
@app.route("/predict_eye", methods=["POST"])
def predict_eye():
    try:
        has_image = "image" in request.files
        symptoms = request.form.get("symptoms", "").strip()

        # ---------------------------
        # ONLY SYMPTOMS (GENERAL MODEL)
        # ---------------------------
        if not has_image and symptoms != "":
            print("Using General Disease Model (Symptoms Only)")

            result = predict_disease(symptoms)
            return jsonify({
                "mode": "symptoms_only_general_disease",
                "result": result
            }), 200

        # ---------------------------
        # ONLY IMAGE (EYE IMAGE MODEL) ✅ OPTION 2
        # ---------------------------
        if has_image and symptoms == "":
            print("Using Eye Image Only Model")

            file = request.files["image"]
            img_bytes = file.read()

            # Validate Eye Image
            valid, msg = is_valid_eye_image(img_bytes)
            if not valid:
                return jsonify({
                    "mode": "eye_validation_failed",
                    "message": msg
                }), 400

            # Save image temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
                temp.write(img_bytes)
                temp_path = temp.name

            try:
                image_pred, image_conf = predict_image_from_bytes(temp_path)
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

            return jsonify({
                "mode": "eye_image_only",
                "image_prediction": {
                    "disease": image_pred,
                    "confidence": image_conf
                }
            }), 200

        # ---------------------------
        # IMAGE + SYMPTOMS (FULL PIPELINE)
        # ---------------------------
        if has_image and symptoms != "":
            print("Using Eye Image + Eye Symptoms Model")

            file = request.files["image"]
            img_bytes = file.read()

            # Validate image
            valid, msg = is_valid_eye_image(img_bytes)
            if not valid:
                return jsonify({
                    "mode": "eye_validation_failed",
                    "message": msg
                }), 400

            # Save temporary image
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
                temp.write(img_bytes)
                temp_path = temp.name

            try:
                image_pred, image_conf = predict_image_from_bytes(temp_path)
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

            # Symptom prediction
            symptoms_list = [s.strip() for s in symptoms.split(",") if s.strip()]
            symptom_pred, symptom_conf, _ = predict_disease_from_symptoms(symptoms_list)

            # Final decision
            decision = final_decision(
                image_pred,
                image_conf,
                symptom_pred,
                symptom_conf
            )

            return jsonify({
                "mode": "eye_image_plus_eye_symptoms",
                "image_prediction": {
                    "disease": image_pred,
                    "confidence": image_conf
                },
                "symptom_prediction": {
                    "disease": symptom_pred,
                    "confidence": symptom_conf
                },
                "final_result": decision
            }), 200

        # ---------------------------
        # NOTHING PROVIDED
        # ---------------------------
        return jsonify({
            "error": "Provide image and/or symptoms"
        }), 400

    except Exception as e:
        print("Exception in /predict_eye:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
