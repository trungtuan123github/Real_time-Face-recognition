import os
import cv2
import face_recognition
import pickle

# Paths
DATASET_DIR = "dataset"
ENCODINGS_PATH = "encodings/encodings.pkl"

# Output lists
known_encodings = []
known_names = []

# Loop through dataset images
for filename in os.listdir(DATASET_DIR):
    if filename.endswith((".jpg", ".jpeg", ".png")):
        path = os.path.join(DATASET_DIR, filename)
        name = os.path.splitext(filename)[0]

        image = cv2.imread(path)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        boxes = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(rgb, boxes)

        for encoding in encodings:
            known_encodings.append(encoding)
            known_names.append(name)

# Save encodings
os.makedirs(os.path.dirname(ENCODINGS_PATH), exist_ok=True)

data = {"encodings": known_encodings, "names": known_names}
with open(ENCODINGS_PATH, "wb") as f:
    pickle.dump(data, f)

print(f"[INFO] Encoded {len(known_encodings)} faces.")
