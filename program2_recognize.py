#!/usr/bin/env python3
"""
Program 2: Face Recognition + Gemini Greeting + Bluetooth TX
-------------------------------------------------------------
• Reads live frames from Raspberry Pi Camera Module 3 (via picamera2)
• Detects faces with OpenCV Haar cascade (fast, no internet)
• Computes ArcFace embedding via DeepFace and compares to known_db
• If cosine similarity ≥ THRESHOLD → calls Gemini API for a greeting
• Sends the greeting string over Bluetooth RFCOMM to the paired Android device

Requirements (install on Pi):
    pip install deepface tf-keras google-generativeai
    sudo apt install -y bluez   # for RFCOMM support in kernel

Uses Python's built-in socket module for Bluetooth — no pybluez needed.

Environment variables (set in ~/.bashrc or a .env file):
    GEMINI_API_KEY=<your key>
    BT_DEVICE_ADDR=XX:XX:XX:XX:XX:XX   # Android Bluetooth MAC
    BT_PORT=1                           # RFCOMM channel (usually 1)
"""

import os
import sys
import json
import time
import socket           # built-in — AF_BLUETOOTH / BTPROTO_RFCOMM
import threading
import queue
import numpy as np
import cv2
from picamera2 import Picamera2
from deepface import DeepFace
import google.generativeai as genai

# ── Configuration ──────────────────────────────────────────────────────────────
DB_FILE         = "face_db.json"
THRESHOLD       = 0.70       # cosine-similarity threshold (0–1; higher = stricter)
MODEL_NAME      = "ArcFace"
COOLDOWN_SEC    = 10         # seconds before the same person can trigger again
FRAME_W, FRAME_H = 1280, 720

GEMINI_API_KEY  = os.environ.get("GEMINI_API_KEY", "")
BT_DEVICE_ADDR  = os.environ.get("BT_DEVICE_ADDR", "")   # e.g. "AA:BB:CC:DD:EE:FF"
BT_PORT         = int(os.environ.get("BT_PORT", "1"))
# ──────────────────────────────────────────────────────────────────────────────


# ── Gemini setup ──────────────────────────────────────────────────────────────
if not GEMINI_API_KEY:
    print("[WARN] GEMINI_API_KEY not set — Gemini calls will fail.")
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")
# ──────────────────────────────────────────────────────────────────────────────


# ── Bluetooth helpers (pure Python socket — no pybluez needed) ────────────────
_bt_socket = None

def bt_connect() -> bool:
    global _bt_socket
    if not BT_DEVICE_ADDR:
        print("[BT] BT_DEVICE_ADDR not set — Bluetooth disabled.")
        return False
    try:
        # AF_BLUETOOTH + BTPROTO_RFCOMM is available in Python 3.3+ on Linux
        sock = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM)
        sock.settimeout(10)
        sock.connect((BT_DEVICE_ADDR, BT_PORT))
        sock.settimeout(None)
        _bt_socket = sock
        print(f"[BT] Connected to {BT_DEVICE_ADDR} on channel {BT_PORT}")
        return True
    except Exception as e:
        print(f"[BT] Connection failed: {e}")
        _bt_socket = None
        return False


def bt_send(text: str):
    global _bt_socket
    if _bt_socket is None:
        print(f"[BT] (not connected) would send: {text!r}")
        return
    try:
        _bt_socket.send((text + "\n").encode("utf-8"))
        print(f"[BT] Sent: {text!r}")
    except Exception as e:
        print(f"[BT] Send error: {e}. Attempting reconnect…")
        _bt_socket = None
        if bt_connect():
            bt_send(text)
# ──────────────────────────────────────────────────────────────────────────────


# ── Face DB ───────────────────────────────────────────────────────────────────
def load_db() -> dict:
    """Returns {name: [embedding_list, …]}"""
    if not os.path.exists(DB_FILE):
        print(f"[DB] {DB_FILE} not found. Run program1_enroll_faces.py first.")
        return {}
    with open(DB_FILE) as f:
        raw = json.load(f)
    # Convert lists back to numpy arrays
    return {name: [np.array(e) for e in embs] for name, embs in raw.items()}


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def identify_face(embedding: np.ndarray, db: dict) -> tuple[str | None, float]:
    """Return (best_name, best_score) or (None, 0.0) if below threshold."""
    best_name, best_score = None, 0.0
    for name, emb_list in db.items():
        for stored_emb in emb_list:
            score = cosine_similarity(embedding, stored_emb)
            if score > best_score:
                best_score = score
                best_name = name
    if best_score >= THRESHOLD:
        return best_name, best_score
    return None, best_score
# ──────────────────────────────────────────────────────────────────────────────


# ── Gemini greeting ───────────────────────────────────────────────────────────
def get_gemini_greeting(name: str) -> str:
    prompt = (
        f"Generate a single short, friendly greeting for someone named {name} "
        f"who was just recognised by a face-recognition door system. "
        f"Examples: '{name} is here!', 'Hey, {name} just arrived — say hi!', "
        f"'Welcome back, {name}!'. Reply with ONLY the greeting phrase, no extra text."
    )
    try:
        response = gemini_model.generate_content(prompt)
        greeting = response.text.strip().strip('"').strip("'")
        print(f"[Gemini] Greeting for {name}: {greeting!r}")
        return greeting
    except Exception as e:
        print(f"[Gemini] Error: {e}")
        return f"{name} is here!"   # graceful fallback
# ──────────────────────────────────────────────────────────────────────────────


# ── Background worker ─────────────────────────────────────────────────────────
# We push (name, score) tuples here; a thread handles Gemini + BT so the
# camera loop never blocks.
_work_queue: queue.Queue = queue.Queue()

def _worker():
    while True:
        item = _work_queue.get()
        if item is None:
            break
        name, score = item
        greeting = get_gemini_greeting(name)
        bt_send(greeting)
        _work_queue.task_done()

_worker_thread = threading.Thread(target=_worker, daemon=True)
_worker_thread.start()
# ──────────────────────────────────────────────────────────────────────────────


# ── Embedding helper ─────────────────────────────────────────────────────────
def get_embedding(image_bgr: np.ndarray) -> np.ndarray | None:
    try:
        result = DeepFace.represent(
            img_path=image_bgr,
            model_name=MODEL_NAME,
            enforce_detection=True,
            detector_backend="opencv",
        )
        if result:
            return np.array(result[0]["embedding"])
    except Exception:
        pass
    return None
# ──────────────────────────────────────────────────────────────────────────────


# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    db = load_db()
    if not db:
        print("[ERROR] Empty face database. Exiting.")
        sys.exit(1)
    print(f"[DB] Loaded {len(db)} person(s): {list(db.keys())}")

    bt_connect()   # attempt once; will auto-retry on send failure

    # Haar cascade for fast face detection (used to crop ROI before DeepFace)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (FRAME_W, FRAME_H), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(1)

    # Cooldown tracker: {name: last_triggered_timestamp}
    last_seen: dict[str, float] = {}

    print("[Main] Recognition running. Press Q in the preview window to quit.\n")

    try:
        while True:
            frame_rgb = picam2.capture_array()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            gray      = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
            )

            for (x, y, w, h) in faces:
                # Slight padding so ArcFace gets the full face
                pad = int(0.15 * max(w, h))
                x1 = max(0, x - pad);  y1 = max(0, y - pad)
                x2 = min(FRAME_W, x + w + pad);  y2 = min(FRAME_H, y + h + pad)
                roi = frame_bgr[y1:y2, x1:x2]

                emb = get_embedding(roi)
                if emb is None:
                    continue

                name, score = identify_face(emb, db)

                if name:
                    now = time.time()
                    if now - last_seen.get(name, 0) >= COOLDOWN_SEC:
                        last_seen[name] = now
                        print(f"[Recog] ✓ {name} (score={score:.3f}) — queuing greeting")
                        _work_queue.put((name, score))
                    label = f"{name} ({score:.2f})"
                    color = (0, 200, 0)
                else:
                    label = f"Unknown ({score:.2f})"
                    color = (0, 0, 220)

                cv2.rectangle(frame_bgr, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame_bgr, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

            cv2.putText(frame_bgr, "Q=quit", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.imshow("Face Recognition", frame_bgr)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[Main] Quit requested.")
                break

    finally:
        picam2.stop()
        cv2.destroyAllWindows()
        _work_queue.put(None)   # signal worker to stop
        _worker_thread.join(timeout=5)
        if _bt_socket:
            _bt_socket.close()
        print("[Main] Shutdown complete.")


if __name__ == "__main__":
    main()
