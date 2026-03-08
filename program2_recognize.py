#!/usr/bin/env python3
"""
Program 2: Face Recognition + Gemini Greeting + WiFi TX
-------------------------------------------------------------
Uses DeepFace ArcFace embeddings compared via EUCLIDEAN DISTANCE.

Threshold: distance < THRESHOLD means "same person".
  Typical good values: 0.9 – 1.2  (start at 1.1, lower if false positives)

Sends greeting strings to the companion Android app over WiFi (TCP port 5050).
The Pi runs a TCP server; the Android app connects as a client.
"""

import os
import sys
import json
import time
import socket
import threading
import queue
import numpy as np
import cv2
from deepface import DeepFace
import google.generativeai as genai

# ── Configuration ──────────────────────────────────────────────────────────────
DB_FILE          = "face_db.json"
THRESHOLD        = 3.5      # euclidean DISTANCE — match if dist < THRESHOLD
MODEL_NAME       = "ArcFace"
COOLDOWN_SEC     = 10
FRAME_W, FRAME_H = 1280, 720

GEMINI_API_KEY   = os.environ.get("GEMINI_API_KEY", "")
TCP_PORT         = int(os.environ.get("TCP_PORT", "5050"))
# ──────────────────────────────────────────────────────────────────────────────


# ── Gemini setup ──────────────────────────────────────────────────────────────
if not GEMINI_API_KEY:
    print("[WARN] GEMINI_API_KEY not set — Gemini calls will fail.")
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")
# ──────────────────────────────────────────────────────────────────────────────


# ── WiFi TCP Server ──────────────────────────────────────────────────────────
_clients: list[socket.socket] = []
_clients_lock = threading.Lock()
_server_socket = None

def start_tcp_server():
    """Start a TCP server that accepts Android client connections."""
    global _server_socket
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", TCP_PORT))
    srv.listen(5)
    _server_socket = srv
    print(f"[WiFi] TCP server listening on port {TCP_PORT}")

    def accept_loop():
        while True:
            try:
                client, addr = srv.accept()
                with _clients_lock:
                    _clients.append(client)
                print(f"[WiFi] Client connected: {addr}")
            except Exception as e:
                print(f"[WiFi] Accept error: {e}")
                break

    t = threading.Thread(target=accept_loop, daemon=True)
    t.start()

def wifi_send(text: str):
    """Send a line of text to all connected Android clients."""
    data = (text + "\n").encode("utf-8")
    with _clients_lock:
        dead = []
        for client in _clients:
            try:
                client.sendall(data)
                print(f"[WiFi] Sent: {text!r}")
            except Exception as e:
                print(f"[WiFi] Send error: {e} — dropping client")
                dead.append(client)
        for d in dead:
            _clients.remove(d)
            try:
                d.close()
            except Exception:
                pass
# ──────────────────────────────────────────────────────────────────────────────


# ── Face DB ───────────────────────────────────────────────────────────────────
def load_db() -> dict:
    if not os.path.exists(DB_FILE):
        print(f"[DB] {DB_FILE} not found. Run program1_enroll_faces.py first.")
        return {}
    with open(DB_FILE) as f:
        raw = json.load(f)
    db = {name: [np.array(e) for e in embs] for name, embs in raw.items()}
    for name, embs in db.items():
        print(f"[DB] Loaded '{name}' — {len(embs)} embedding(s), "
              f"dim={embs[0].shape if embs else 'N/A'}")
    return db


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """L2 distance — correct metric for ArcFace embeddings."""
    return float(np.linalg.norm(a - b))


def identify_face(embedding: np.ndarray, db: dict) -> tuple:
    """
    Returns (best_name, best_distance) if best_distance < THRESHOLD,
    otherwise (None, best_distance).
    Lower distance = more similar.
    """
    best_name, best_dist = None, float("inf")
    for name, emb_list in db.items():
        for stored_emb in emb_list:
            dist = euclidean_distance(embedding, stored_emb)
            if dist < best_dist:
                best_dist = dist
                best_name = name

    print(f"  [identify] best={best_name}  dist={best_dist:.4f}  "
          f"({'MATCH' if best_dist < THRESHOLD else 'no match'})")

    if best_dist < THRESHOLD:
        return best_name, best_dist
    return None, best_dist
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
        return f"{name} is here!"
# ──────────────────────────────────────────────────────────────────────────────


# ── Background worker ─────────────────────────────────────────────────────────
_work_queue: queue.Queue = queue.Queue()

def _worker():
    while True:
        item = _work_queue.get()
        if item is None:
            break
        name, dist = item
        greeting = get_gemini_greeting(name)
        wifi_send(greeting)
        _work_queue.task_done()

_worker_thread = threading.Thread(target=_worker, daemon=True)
_worker_thread.start()
# ──────────────────────────────────────────────────────────────────────────────


# ── Embedding helper ──────────────────────────────────────────────────────────
def get_faces(image_bgr: np.ndarray) -> list[dict]:
    """
    Detect all faces in the full frame and return a list of dicts with keys:
      - embedding: np.ndarray
      - facial_area: dict with x, y, w, h
    """
    try:
        results = DeepFace.represent(
            img_path=image_bgr,
            model_name=MODEL_NAME,
            enforce_detection=True,
            detector_backend="opencv",
        )
        faces = []
        for r in results:
            faces.append({
                "embedding": np.array(r["embedding"]),
                "facial_area": r["facial_area"],
            })
        if faces:
            print(f"  [detect] {len(faces)} face(s), "
                  f"dim={faces[0]['embedding'].shape}")
        return faces
    except Exception as e:
        print(f"  [detect] failed: {e}")
    return []
# ──────────────────────────────────────────────────────────────────────────────


# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    from picamera2 import Picamera2

    db = load_db()
    if not db:
        print("[ERROR] Empty face database. Exiting.")
        sys.exit(1)
    print(f"\n[Config] People: {list(db.keys())}")
    print(f"[Config] Metric: euclidean distance  |  Match if dist < {THRESHOLD}\n")

    start_tcp_server()

    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (FRAME_W, FRAME_H), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(1)

    last_seen: dict[str, float] = {}
    print("[Main] Recognition running. Press Q to quit.\n")

    try:
        while True:
            frame_rgb = picam2.capture_array()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            detected_faces = get_faces(frame_bgr)

            for face in detected_faces:
                emb = face["embedding"]
                fa = face["facial_area"]
                x, y, w, h = fa["x"], fa["y"], fa["w"], fa["h"]

                name, dist = identify_face(emb, db)

                if name:
                    now = time.time()
                    if now - last_seen.get(name, 0) >= COOLDOWN_SEC:
                        last_seen[name] = now
                        print(f"[Recog] ✓ MATCH: {name} dist={dist:.4f} — queuing greeting")
                        _work_queue.put((name, dist))
                    label = f"{name} ({dist:.2f})"
                    color = (0, 200, 0)
                else:
                    label = f"Unknown ({dist:.2f})"
                    color = (0, 0, 220)

                cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame_bgr, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

            cv2.putText(frame_bgr, f"dist<{THRESHOLD} to match  |  Q=quit", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
            cv2.imshow("Face Recognition", frame_bgr)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[Main] Quit.")
                break

    finally:
        picam2.stop()
        cv2.destroyAllWindows()
        _work_queue.put(None)
        _worker_thread.join(timeout=5)
        with _clients_lock:
            for c in _clients:
                c.close()
        if _server_socket:
            _server_socket.close()
        print("[Main] Shutdown complete.")


if __name__ == "__main__":
    main()