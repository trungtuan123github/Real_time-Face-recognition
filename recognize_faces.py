import os
import sys
import cv2
import pickle
import numpy as np
import face_recognition
from scipy.spatial.distance import cosine

# Configuration
ENCODINGS_PATH = 'encodings/encodings.pkl'
THRESHOLD = 0.2  # Lower = more similar

# Validate encoding file exists
if not os.path.isfile(ENCODINGS_PATH):
    print(f"[ERROR] Encodings file not found at '{ENCODINGS_PATH}'.")
    sys.exit(1)

# Load encoded face data
with open(ENCODINGS_PATH, 'rb') as f:
    data = pickle.load(f)

known_encodings = data.get('encodings', [])
known_names = data.get('names', [])

if not known_encodings or not known_names:
    print("[ERROR] No face encodings or names found. Please run the encoding script first.")
    sys.exit(1)

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Could not access webcam.")
    sys.exit(1)

print("[INFO] Real-time face recognition started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to read frame.")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)

    if not face_locations:
        cv2.imshow("Real-time Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        label = "Unknown"
        similarity = 0.0

        distances = [cosine(face_encoding, known) for known in known_encodings]
        if distances:
            best_match_index = np.argmin(distances)
            best_score = distances[best_match_index]
            similarity = 1 - best_score

            if best_score < THRESHOLD:
                label = known_names[best_match_index]

        # Draw bounding box and label
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} (sim: {similarity:.2f})", (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Real-time Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("[INFO] Program terminated.")