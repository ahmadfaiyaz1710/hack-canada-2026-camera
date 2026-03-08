"""
face_api_server.py — HTTP REST API for face enrollment (port 5000).

Called by the SmartCane companion app (WiFiService.ts) to manage registered
faces. Processing (DeepFace embedding) runs in a background thread so the
HTTP response returns immediately.

Endpoints:
  GET  /ping                → {"ok": true}
  GET  /faces/list          → [{name, relationship, addedAt}, ...]
  POST /faces/register      → multipart: name, relationship, photo_0..N
  DELETE /faces/<name>      → remove person from DB
  GET  /faces/status        → {"status": "idle"|"processing"|"done"|"error"}
"""

import json
import logging
import os
import threading
import time

import cv2
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

logger = logging.getLogger(__name__)

DATA_DIR  = os.path.join(os.path.dirname(__file__), '..', 'data')
DB_FILE   = os.path.join(DATA_DIR, 'face_db.json')
FACES_DIR = os.path.join(DATA_DIR, 'known_faces')
MODEL_NAME = 'ArcFace'

app = Flask(__name__)
CORS(app)

_encoding_status: str = 'idle'   # idle | processing | done | error
_on_db_updated = None             # callback → face_recognizer.reload_db
_START_TIME = time.time()


# ── DB helpers ────────────────────────────────────────────────────────────────

def _load_db() -> dict:
    if not os.path.exists(DB_FILE):
        return {}
    with open(DB_FILE) as f:
        return json.load(f)


def _save_db(db: dict):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(DB_FILE, 'w') as f:
        json.dump(db, f, indent=2)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/ping')
def ping():
    return jsonify({'ok': True})


@app.route('/faces/list')
def faces_list():
    db = _load_db()
    result = [
        {
            'name':         name,
            'relationship': data.get('relationship', ''),
            'addedAt':      data.get('addedAt', ''),
        }
        for name, data in db.items()
    ]
    return jsonify(result)


@app.route('/faces/register', methods=['POST'])
def faces_register():
    global _encoding_status

    name         = request.form.get('name', '').strip()
    relationship = request.form.get('relationship', 'Other').strip()

    if not name:
        return jsonify({'success': False, 'error': 'Name is required'}), 400

    # Decode uploaded photos from multipart form
    photos = []
    for key in sorted(request.files.keys()):
        if key.startswith('photo_'):
            raw = request.files[key].read()
            arr = np.frombuffer(raw, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None:
                photos.append(img)

    if not photos:
        return jsonify({'success': False, 'error': 'No valid photos received'}), 400

    _encoding_status = 'processing'
    encoding_id = str(int(time.time()))

    threading.Thread(
        target=_run_enrollment,
        args=(name, relationship, photos),
        daemon=True,
    ).start()

    return jsonify({'success': True, 'encodingId': encoding_id})


@app.route('/faces/<name>', methods=['DELETE'])
def faces_delete(name: str):
    db = _load_db()
    if name not in db:
        return jsonify({'ok': False, 'error': 'Not found'}), 404
    del db[name]
    _save_db(db)
    _notify_db_updated()
    logger.info(f'[FaceAPI] Deleted face: {name}')
    return jsonify({'ok': True})


@app.route('/faces/status')
def faces_status():
    return jsonify({'status': _encoding_status})


@app.route('/status')
def device_status():
    """Returns Pi device info for the companion app to poll."""
    db = _load_db()
    battery = _read_battery()
    return jsonify({
        'battery':      battery,
        'faces_count':  len(db),
        'uptime':       round(time.time() - _START_TIME),
    })


def _read_battery() -> int:
    """Read battery % from UPS HAT if available, otherwise return 100."""
    try:
        # INA219-based UPS HATs expose voltage via /sys or I2C
        with open('/sys/class/power_supply/BAT0/capacity') as f:
            return int(f.read().strip())
    except Exception:
        return 100  # no UPS HAT — report full


# ── Background enrollment ─────────────────────────────────────────────────────

def _run_enrollment(name: str, relationship: str, photos: list):
    global _encoding_status
    try:
        from deepface import DeepFace

        person_dir = os.path.join(
            FACES_DIR, name.lower().replace(' ', '_')
        )
        os.makedirs(person_dir, exist_ok=True)

        db = _load_db()
        existing = []  # always replace embeddings on re-enroll
        new_embeddings = []

        for i, img_bgr in enumerate(photos):
            # Save reference image
            idx = len(existing) + i + 1
            img_path = os.path.join(person_dir, f'{name}_{idx:03d}.jpg')
            cv2.imwrite(img_path, img_bgr)

            # Extract ArcFace embedding
            try:
                result = DeepFace.represent(
                    img_path=img_bgr,
                    model_name=MODEL_NAME,
                    enforce_detection=True,
                    detector_backend='opencv',
                )
                if result:
                    new_embeddings.append(result[0]['embedding'])
                    logger.debug(f'[FaceAPI] Embedded photo {idx} for {name}')
            except Exception as e:
                logger.warning(f'[FaceAPI] Photo {i} embedding failed: {e}')

        if not new_embeddings:
            logger.error(f'[FaceAPI] No embeddings generated for {name}')
            _encoding_status = 'error'
            return

        db[name] = {
            'relationship': relationship,
            'addedAt':      time.strftime('%Y-%m-%d'),
            'embeddings':   existing + new_embeddings,
        }
        _save_db(db)
        _notify_db_updated()

        logger.info(
            f'[FaceAPI] Enrolled {name} — {len(new_embeddings)} new embeddings '
            f'({len(existing) + len(new_embeddings)} total)'
        )
        _encoding_status = 'done'

    except Exception as e:
        logger.error(f'[FaceAPI] Enrollment error: {e}')
        _encoding_status = 'error'


def _notify_db_updated():
    if _on_db_updated:
        try:
            _on_db_updated()
        except Exception as e:
            logger.warning(f'[FaceAPI] DB updated callback error: {e}')


# ── Public API ────────────────────────────────────────────────────────────────

def set_db_updated_callback(callback):
    """Register a callback (e.g. face_recognizer.reload_db) to run after enrollment."""
    global _on_db_updated
    _on_db_updated = callback


def run(host: str = '0.0.0.0', port: int = 5000):
    """Start the Flask server (blocking). Run in its own thread from main.py."""
    logger.info(f'[FaceAPI] Starting HTTP server on {host}:{port}')
    # Use werkzeug directly to suppress Flask's startup banner noise
    from werkzeug.serving import make_server
    server = make_server(host, port, app)
    server.serve_forever()
