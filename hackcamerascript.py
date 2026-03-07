"""
Program 1: Face Capture
=======================
Captures and saves face images + encodings using Pi Camera Module 3.
Run this once per person you want to register.

Install dependencies:
    sudo apt install python3-opencv python3-picamera2
    pip install face_recognition numpy
"""

import os
import cv2
import numpy as np
import face_recognition
import pickle
from picamera2 import Picamera2
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────
FACES_DIR   = "known_faces"          # folder to save face images
ENCODINGS_FILE = "face_encodings.pkl"  # saved encodings for program 2
CAPTURE_COUNT  = 10                  # how many photos to take per person
# ─────────────────────────────────────────────────────────────────────────────

def setup_dirs(name: str) -> str:
    person_dir = os.path.join(FACES_DIR, name)
    os.makedirs(person_dir, exist_ok=True)
    return person_dir


def load_existing_encodings() -> dict:
    if os.path.exists(ENCODINGS_FILE):
        with open(ENCODINGS_FILE, "rb") as f:
            return pickle.load(f)
    return {}


def save_encodings(encodings: dict):
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump(encodings, f)
    print(f"[✓] Encodings saved to {ENCODINGS_FILE}")


def capture_faces(name: str):
    person_dir = setup_dirs(name)
    encodings_db = load_existing_encodings()

    if name not in encodings_db:
        encodings_db[name] = []

    # Init camera
    cam = Picamera2()
    config = cam.create_preview_configuration(main={"size": (1280, 720)})
    cam.configure(config)
    cam.start()

    print(f"\n[→] Registering: {name}")
    print("    Press SPACE to capture a face, Q to quit early.\n")

    captured = 0
    cv2_window = "Face Capture — Press SPACE to capture, Q to quit"

    while captured < CAPTURE_COUNT:
        frame = cam.capture_array()
        display = frame.copy()

        # Detect faces for preview overlay
        small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        face_locs = face_recognition.face_locations(rgb_small)

        for (top, right, bottom, left) in face_locs:
            top, right, bottom, left = top*4, right*4, bottom*4, left*4
            cv2.rectangle(display, (left, top), (right, bottom), (0, 255, 0), 2)

        status = f"Captured: {captured}/{CAPTURE_COUNT}  |  Faces detected: {len(face_locs)}"
        cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)

        cv2.imshow(cv2_window, display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("[!] Quit early.")
            break

        if key == ord(' '):
            if not face_locs:
                print("[!] No face detected — try again.")
                continue

            # Use the first detected face
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(rgb_frame, face_locs)

            if encodings:
                encodings_db[name].append(encodings[0])

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                img_path = os.path.join(person_dir, f"{name}_{timestamp}.jpg")
                cv2.imwrite(img_path, frame)

                captured += 1
                print(f"[✓] Captured {captured}/{CAPTURE_COUNT} — saved to {img_path}")

    cam.stop()
    cv2.destroyAllWindows()

    save_encodings(encodings_db)
    print(f"\n[✓] Done! {captured} images saved for '{name}'.")
    print(f"    Total people registered: {list(encodings_db.keys())}\n")


if __name__ == "__main__":
    print("=== Face Registration ===")
    name = input("Enter person's name: ").strip()
    if not name:
        print("[!] Name cannot be empty.")
    else:
        capture_faces(name)