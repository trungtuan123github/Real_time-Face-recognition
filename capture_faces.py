import cv2
import face_recognition
import os

SAVE_DIR = 'dataset'
os.makedirs(SAVE_DIR, exist_ok=True)

cap = cv2.VideoCapture(0)
index = 0

print("Press 's' to save detected face | 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect face locations
    face_locations = face_recognition.face_locations(frame)

    for top, right, bottom, left in face_locations:
        # Draw box
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

    cv2.imshow("Capture Faces", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        for face_location in face_locations:
            top, right, bottom, left = face_location
            face_image = frame[top:bottom, left:right]
            path = os.path.join(SAVE_DIR, f"{index}.jpg")
            cv2.imwrite(path, face_image)
            print(f"Saved {path}")
            index += 1
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
