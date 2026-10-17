"""
capture_extra_photo.py
-----------------------
Run this script to capture an extra reference photo for a student
(e.g., without glasses). The captured image is saved as:
    dataset/<rollno>_noglasses.jpg

Usage:
    python capture_extra_photo.py

Press SPACE to capture, ESC to cancel.
"""

import cv2
import os
import face_recognition

DATASET_PATH = "dataset"
ROLL_NO = "100"          # <-- change this to the student's roll number
LABEL = "noglasses"      # <-- change the label if you want (e.g. 'alt', 'side')

save_path = os.path.join(DATASET_PATH, f"{ROLL_NO}_{LABEL}.jpg")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print(f"[INFO] Looking at camera... Press SPACE to capture, ESC to cancel.")
print(f"[INFO] Will save to: {save_path}")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    locs = face_recognition.face_locations(rgb, number_of_times_to_upsample=1, model="hog")

    display = frame.copy()
    if locs:
        for top, right, bottom, left in locs:
            cv2.rectangle(display, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(display, f"Face detected! SPACE to capture", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    else:
        cv2.putText(display, "No face – move closer", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow(f"Capture extra photo for Roll {ROLL_NO} – press SPACE", display)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        print("[INFO] Cancelled.")
        break
    elif key == 32 and locs:  # SPACE
        cv2.imwrite(save_path, frame)
        print(f"[OK] Saved: {save_path}")

        # Verify encoding works
        img = face_recognition.load_image_file(save_path)
        encs = face_recognition.face_encodings(img, num_jitters=5, model="small")
        if encs:
            print(f"[OK] Face encoding successful — {len(encs)} encoding(s) found.")
            print(f"[DONE] Restart the attendance system to load the new reference photo.")
        else:
            print("[WARNING] Face encoding failed on saved image. Try again with better lighting.")
            os.remove(save_path)
        break

cap.release()
cv2.destroyAllWindows()
