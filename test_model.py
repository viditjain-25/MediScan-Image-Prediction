import requests

# -----------------------------
# CONFIG
# -----------------------------
# Local Flask URLs
SYMPTOM_URL = "http://127.0.0.1:5000/predict"
EYE_URL = "http://127.0.0.1:5000/predict_eye"

# Image path
IMAGE_PATH = r"C:\Users\Divy\Desktop\download.jpg"

# Symptoms for testing
SYMPTOMS = "red_eye, blurred_vision"

# -----------------------------
# 1️⃣ Test symptoms-only
# -----------------------------
print("Testing symptoms-only request...")
resp = requests.post(SYMPTOM_URL, json={"symptoms": SYMPTOMS})
print("Response:")
print(resp.json())
print("-" * 50)

# -----------------------------
# 2️⃣ Test image + symptoms
# -----------------------------
print("Testing eye image + symptoms request...")
with open(IMAGE_PATH, "rb") as f:
    files = {"image": f}
    data = {"symptoms": SYMPTOMS}
    resp = requests.post(EYE_URL, files=files, data=data)

print("Response:")
print(resp.json())
