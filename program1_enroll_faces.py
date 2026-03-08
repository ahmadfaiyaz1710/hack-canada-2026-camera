#!/usr/bin/env python3
"""
Program 1: Face Enrollment
--------------------------
Captures face images using Raspberry Pi Camera Module 3,
extracts embeddings using DeepFace (ArcFace model),
and saves them to a local database (JSON + images folder).

Usage:
    python3 program1_enroll_faces.py --name "Ahmad"
    python3 program1_enroll_faces.py --name "Ahmad" --samples 10
"""

import argparse
import os
import sys
import json
import time
import numpy as np
import cv2
from picamera2 import Picamera2
from deepface import DeepFace

# ── Configuration ──────────────────────────────────────────────────────────────
FACES_DIR       = "known_faces"          # root folder for saved face images
DB_FILE         = "face_db.json"         # maps name → list of embedding vectors
DEFAULT_SAMPLES = 8                      # how many images to capture per person
CAPTURE_DELAY   = 0.8                    # seconds between auto-captures
FRAME_W, FRAME_H = 1280, 720
MODEL_NAME      = "ArcFace"              # DeepFace model used for embeddings
# ──────────────────────────────────────────────────────────────────────────────


def load_db() -> dict:
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f:
            return json.load(f)
    return {}


def save_db(db: dict):
    with open(DB_FILE, "w") as f:
        json.dump(db, f, indent=2)
    print(f"[DB] Saved database → {DB_FILE}")


def get_embedding(image_bgr: np.ndarray) -> list | None:
    """Return ArcFace embedding list for the largest face in the image, or None."""
    try:
        result = DeepFace.represent(
            img_path=image_bgr,
            model_name=MODEL_NAME,
            enforce_detection=True,
            detector_backend="opencv",   # fast; swap to "retinaface" for accuracy
        )
        if result:
            return result[0]["embedding"]
    except Exception as e:
        print(f"  [embed] No face detected or error: {e}")
    return None


def capture_and_enroll(name: str, n_samples: int):
    os.makedirs(FACES_DIR, exist_ok=True)
    person_dir = os.path.join(FACES_DIR, name.lower().replace(" ", "_"))
    os.makedirs(person_dir, exist_ok=True)

    db = load_db()
    embeddings_for_person = db.get(name, [])

    print(f"\n[Enroll] Starting enrollment for: {name}")
    print(f"         Will capture {n_samples} samples.")
    print("         Press SPACE to manually capture, Q to quit early.\n")

    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (FRAME_W, FRAME_H), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(1)  # warm-up

    captured = 0
    last_auto_time = time.time()

    try:
        while captured < n_samples:
            frame_rgb = picam2.capture_array()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # Draw overlay
            cv2.putText(
                frame_bgr,
                f"Enrolling: {name}  [{captured}/{n_samples}]",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 0), 2,
            )
            cv2.putText(
                frame_bgr,
                "SPACE=capture  Q=quit",
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 1,
            )
            cv2.imshow("Enroll Face", frame_bgr)

            key = cv2.waitKey(1) & 0xFF
            now = time.time()
            do_capture = (key == ord(" ")) or (now - last_auto_time >= CAPTURE_DELAY)

            if key == ord("q"):
                print("[Enroll] Quit requested.")
                break

            if do_capture:
                last_auto_time = now
                emb = get_embedding(frame_bgr)
                if emb is not None:
                    captured += 1
                    img_path = os.path.join(person_dir, f"{name}_{captured:03d}.jpg")
                    cv2.imwrite(img_path, frame_bgr)
                    embeddings_for_person.append(emb)
                    print(f"  ✓ Sample {captured}/{n_samples} saved → {img_path}")
                else:
                    print("  ✗ No face found in frame, skipping.")
        picam2.stop()
        picam2.close()
        
    finally:
        picam2.stop()
        picam2.close()
        cv2.destroyAllWindows()

    if captured == 0:
        print("[Enroll] No samples captured. Aborting.")
        return

    db[name] = embeddings_for_person
    save_db(db)
    print(f"\n[Enroll] Done! {captured} embeddings stored for '{name}'.")
    print(f"[Enroll] Total people in DB: {list(db.keys())}")


def list_enrolled():
    db = load_db()
    if not db:
        print("No faces enrolled yet.")
        return
    print("\nEnrolled faces:")
    for name, embs in db.items():
        print(f"  • {name}: {len(embs)} embedding(s)")


def delete_person(name: str):
    db = load_db()
    if name in db:
        del db[name]
        save_db(db)
        print(f"[DB] Removed '{name}' from database.")
    else:
        print(f"[DB] '{name}' not found.")


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enroll faces into the local DB.")
    parser.add_argument("--name",    type=str, help="Person's name to enroll")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES,
                        help=f"Number of capture samples (default {DEFAULT_SAMPLES})")
    parser.add_argument("--list",    action="store_true", help="List enrolled people")
    parser.add_argument("--delete",  type=str, metavar="NAME", help="Delete a person from DB")
    args = parser.parse_args()

    if args.list:
        list_enrolled()
    elif args.delete:
        delete_person(args.delete)
    elif args.name:
        capture_and_enroll(args.name, args.samples)
    else:
        parser.print_help()
