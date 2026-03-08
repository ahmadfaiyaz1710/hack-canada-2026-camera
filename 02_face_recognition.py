#!/usr/bin/env python3
"""
Program 2: Face Recognition + Gemini + ElevenLabs TTS
- Continuously scans camera feed for known faces using MediaPipe + face_recognition
- On match (confidence >= 0.7), calls Gemini API for a greeting phrase
- Calls ElevenLabs API to synthesize TTS audio
- Plays audio through MAX98357A I2S speaker
- Cooldown per person to avoid repeated greetings
- Logs all detections to a CSV file

Requirements:
    pip install mediapipe face_recognition picamera2 google-generativeai \
                elevenlabs opencv-python numpy requests

Environment variables (or edit CONFIG below):
    GEMINI_API_KEY
    ELEVENLABS_API_KEY
    ELEVENLABS_VOICE_ID  (optional, defaults to "Rachel")
"""

import cv2
import os
import time
import csv
import logging
import tempfile
import threading
import subprocess
import numpy as np
from datetime import datetime
from pathlib import Path

import mediapipe as mp
import face_recognition
from picamera2 import Picamera2
import google.generativeai as genai
from elevenlabs.client import ElevenLabs
from elevenlabs import play, save

# ─────────────────────────────────────────────
#  CONFIG  –  edit these or use env variables
# ─────────────────────────────────────────────
CONFIG = {
    "gemini_api_key":      os.getenv("GEMINI_API_KEY",      "YOUR_GEMINI_API_KEY"),
    "elevenlabs_api_key":  os.getenv("ELEVENLABS_API_KEY",  "YOUR_ELEVENLABS_API_KEY"),
    "elevenlabs_voice_id": os.getenv("ELEVENLABS_VOICE_ID", "Rachel"),   # or your custom voice ID
    "known_faces_dir":     "known_faces",          # folder with sub-dirs per person
    "recognition_threshold": 0.70,                 # 0.0–1.0  (lower = stricter match)
    "cooldown_seconds":    30,                     # seconds before re-greeting same person
    "log_file":            "face_log.csv",
    "camera_resolution":   (1280, 720),
    "scan_every_n_frames": 5,                      # process every Nth frame for speed
}
# ─────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ── Logging ──────────────────────────────────
def init_log():
    if not os.path.exists(CONFIG["log_file"]):
        with open(CONFIG["log_file"], "w", newline="") as f:
            csv.writer(f).writerow(["timestamp", "name", "confidence", "greeting"])

def log_detection(name: str, confidence: float, greeting: str):
    with open(CONFIG["log_file"], "a", newline="") as f:
        csv.writer(f).writerow([datetime.now().isoformat(), name, f"{confidence:.3f}", greeting])


# ── Face database ─────────────────────────────
def load_known_faces(faces_dir: str):
    """Load all reference images and encode them."""
    known_encodings = []
    known_names = []
    faces_path = Path(faces_dir)

    if not faces_path.exists():
        log.error(f"known_faces/ directory not found. Run 01_capture_faces.py first.")
        return [], []

    for person_dir in faces_path.iterdir():
        if not person_dir.is_dir():
            continue
        name = person_dir.name
        count = 0
        for img_path in person_dir.glob("*.jpg"):
            img = face_recognition.load_image_file(str(img_path))
            encs = face_recognition.face_encodings(img)
            if encs:
                known_encodings.append(encs[0])
                known_names.append(name)
                count += 1
        log.info(f"  Loaded {count} encodings for: {name}")

    log.info(f"✅ Total known faces loaded: {len(known_names)}")
    return known_encodings, known_names


# ── MediaPipe face detector ───────────────────
def build_face_detector():
    mp_face = mp.solutions.face_detection
    return mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5)


# ── Gemini greeting ───────────────────────────
def get_gemini_greeting(name: str) -> str:
    try:
        genai.configure(api_key=CONFIG["gemini_api_key"])
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = (
            f"You are a friendly smart home assistant. {name} has just been detected by a camera. "
            f"Generate a single short, warm, natural greeting phrase (max 12 words) acknowledging their presence. "
            f"Examples: '{name} is here!', 'Hey {name}, welcome back!', '{name} just walked in, say hi?'. "
            f"Only return the phrase itself, nothing else."
        )
        response = model.generate_content(prompt)
        greeting = response.text.strip().strip('"').strip("'")
        log.info(f"Gemini greeting: {greeting}")
        return greeting
    except Exception as e:
        log.error(f"Gemini API error: {e}")
        return f"{name} is here!"   # fallback


# ── ElevenLabs TTS ────────────────────────────
def speak(text: str):
    """Convert text to speech and play via MAX98357A (ALSA/I2S)."""
    try:
        client = ElevenLabs(api_key=CONFIG["elevenlabs_api_key"])
        audio = client.text_to_speech.convert(
            voice_id=CONFIG["elevenlabs_voice_id"],
            text=text,
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128",
        )
        # Save to temp file and play with aplay / mpg123
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = tmp.name
            for chunk in audio:
                tmp.write(chunk)

        # Use mpg123 for MP3 playback on ALSA (MAX98357A shows up as hw:0,0 or similar)
        subprocess.run(["mpg123", "-q", tmp_path], check=True)
        os.unlink(tmp_path)
    except Exception as e:
        log.error(f"ElevenLabs/audio error: {e}")


# ── Main recognition loop ─────────────────────
def run():
    init_log()
    log.info("Loading known faces…")
    known_encodings, known_names = load_known_faces(CONFIG["known_faces_dir"])
    if not known_encodings:
        log.error("No face encodings found. Exiting.")
        return

    detector = build_face_detector()

    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": CONFIG["camera_resolution"], "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(2)
    log.info("📷 Camera started. Scanning…")

    cooldowns: dict[str, float] = {}   # name → last greeted timestamp
    frame_count = 0
    tts_thread: threading.Thread | None = None

    try:
        while True:
            frame_rgb = picam2.capture_array()   # RGB
            frame_count += 1

            if frame_count % CONFIG["scan_every_n_frames"] != 0:
                continue

            # ── MediaPipe: detect face regions ──
            results = detector.process(frame_rgb)
            if not results.detections:
                continue

            h, w = frame_rgb.shape[:2]
            face_locations = []
            for det in results.detections:
                bbox = det.location_data.relative_bounding_box
                top    = max(0, int(bbox.ymin * h))
                left   = max(0, int(bbox.xmin * w))
                bottom = min(h, int((bbox.ymin + bbox.height) * h))
                right  = min(w, int((bbox.xmin + bbox.width) * w))
                face_locations.append((top, right, bottom, left))  # face_recognition order

            # ── face_recognition: encode & compare ──
            small = cv2.resize(frame_rgb, (0, 0), fx=0.5, fy=0.5)
            scaled_locs = [(t//2, r//2, b//2, l//2) for t, r, b, l in face_locations]
            encodings = face_recognition.face_encodings(small, scaled_locs)

            for enc, (top, right, bottom, left) in zip(encodings, face_locations):
                distances = face_recognition.face_distance(known_encodings, enc)
                if len(distances) == 0:
                    continue

                best_idx = int(np.argmin(distances))
                confidence = 1.0 - float(distances[best_idx])

                if confidence < CONFIG["recognition_threshold"]:
                    continue   # unknown / uncertain face — ignore

                name = known_names[best_idx]
                now = time.time()

                # ── Cooldown check ──
                last_seen = cooldowns.get(name, 0)
                if now - last_seen < CONFIG["cooldown_seconds"]:
                    continue

                cooldowns[name] = now
                log.info(f"✅ Recognized: {name}  (confidence={confidence:.2f})")

                # ── Gemini + TTS in background thread ──
                if tts_thread is None or not tts_thread.is_alive():
                    def greet(n=name, c=confidence):
                        greeting = get_gemini_greeting(n)
                        log_detection(n, c, greeting)
                        speak(greeting)

                    tts_thread = threading.Thread(target=greet, daemon=True)
                    tts_thread.start()

    except KeyboardInterrupt:
        log.info("Stopped by user.")
    finally:
        picam2.stop()
        cv2.destroyAllWindows()
        log.info("Shutdown complete.")


if __name__ == "__main__":
    run()
