#!/usr/bin/env python3
"""
Program 1: Face Reference Photo Capture
Captures and saves reference photos for each person to be recognized.
Run this once per person to build your reference database.

Usage:
    python3 01_capture_faces.py --name "Ahmad" --photos 5
"""

import cv2
import os
import argparse
import time
from picamera2 import Picamera2

FACES_DIR = "known_faces"

def capture_reference_photos(name: str, num_photos: int = 5):
    os.makedirs(FACES_DIR, exist_ok=True)
    person_dir = os.path.join(FACES_DIR, name)
    os.makedirs(person_dir, exist_ok=True)

    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (1280, 720), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(2)  # Warm up camera

    print(f"\n📸 Capturing {num_photos} reference photos for: {name}")
    print("Press SPACE to capture, Q to quit early.\n")

    captured = 0
    preview_win = f"Capture - {name} ({captured}/{num_photos})"

    while captured < num_photos:
        frame = picam2.capture_array()
        display = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        cv2.putText(display, f"Person: {name}  |  Captured: {captured}/{num_photos}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(display, "SPACE = capture  |  Q = quit",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        cv2.imshow(preview_win, display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            filename = os.path.join(person_dir, f"{name}_{captured + 1:03d}.jpg")
            # Save original RGB for consistency
            cv2.imwrite(filename, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            captured += 1
            print(f"  ✅ Saved: {filename}")
            time.sleep(0.3)  # Small delay to avoid duplicate frames
        elif key == ord('q'):
            print("  ⚠️  Quit early.")
            break

    picam2.stop()
    cv2.destroyAllWindows()
    print(f"\n✅ Done! {captured} photos saved to: {person_dir}/")
    print("Run 01_capture_faces.py again for each additional person.")
    print("When done, run 02_face_recognition.py to start the recognition system.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture reference face photos.")
    parser.add_argument("--name", required=True, help="Person's name (e.g. Ahmad)")
    parser.add_argument("--photos", type=int, default=5, help="Number of photos to capture (default: 5)")
    args = parser.parse_args()

    capture_reference_photos(args.name, args.photos)
