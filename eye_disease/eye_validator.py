import cv2
import numpy as np

def is_valid_eye_image(file_bytes):
    # Convert bytes → image
    npimg = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if img is None:
        return False, "Invalid image or unreadable image"

    h, w, _ = img.shape

    if h < 100 or w < 100:
        return False, "Image too small"

    return True, "Valid eye image"
