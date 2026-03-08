"""
face_recognizer.py — DeepFace/ArcFace face recognition module.

Called by camera_thread in main.py via recognizer.detect().
Loads face embeddings from data/face_db.json and matches against
live camera frames using Euclidean distance (correct for ArcFace).
"""

import json
import logging
import os
import time

import numpy as np

logger = logging.getLogger(__name__)

DATA_DIR  = os.path.join(os.path.dirname(__file__), '..', 'data')
DB_FILE   = os.path.join(DATA_DIR, 'face_db.json')
THRESHOLD = 3.5        # euclidean distance — match if dist < THRESHOLD
MODEL_NAME = 'ArcFace'
FRAME_W, FRAME_H = 1280, 720


class FaceRecognizer:
    def __init__(self):
        self._db: dict = {}
        self._camera = None
        self._deepface = None
        self._cv2 = None
        self._reload_db()
        self._init_camera()
        self._init_deepface()

    # ── DB ────────────────────────────────────────────────────────────────────

    def _reload_db(self):
        """Load face embeddings from face_db.json. Safe to call at any time."""
        if not os.path.exists(DB_FILE):
            logger.warning('[FaceRecognizer] face_db.json not found — no faces loaded')
            self._db = {}
            return
        try:
            with open(DB_FILE) as f:
                raw = json.load(f)
            db = {}
            for name, data in raw.items():
                db[name] = {
                    'relationship': data.get('relationship', ''),
                    'addedAt':      data.get('addedAt', ''),
                    'embeddings':   [np.array(e) for e in data.get('embeddings', [])],
                }
            self._db = db
            logger.info(f'[FaceRecognizer] Loaded {len(db)} people: {list(db.keys())}')
        except Exception as e:
            logger.error(f'[FaceRecognizer] DB load error: {e}')

    def reload_db(self):
        """Public reload — called by face_api_server after a new enrollment."""
        self._reload_db()

    # ── Hardware init ─────────────────────────────────────────────────────────

    def _init_camera(self):
        try:
            from picamera2 import Picamera2
            cam = Picamera2()
            cfg = cam.create_preview_configuration(
                main={'size': (FRAME_W, FRAME_H), 'format': 'RGB888'}
            )
            cam.configure(cfg)
            cam.start()
            time.sleep(1)
            self._camera = cam
            logger.info('[FaceRecognizer] Camera ready')
        except Exception as e:
            logger.warning(f'[FaceRecognizer] Camera init failed: {e}')

    def _init_deepface(self):
        try:
            from deepface import DeepFace
            import cv2
            self._deepface = DeepFace
            self._cv2 = cv2
            logger.info('[FaceRecognizer] DeepFace ready')
        except ImportError as e:
            logger.warning(f'[FaceRecognizer] DeepFace not available: {e}')

    # ── Main API ──────────────────────────────────────────────────────────────

    def detect(self) -> list[dict]:
        """
        Capture one frame, run ArcFace, return matches.
        Returns: list of {'name': str, 'confidence': float, 'relationship': str}
        """
        if not self._camera or not self._deepface or not self._cv2:
            return []
        if not self._db:
            return []

        try:
            frame_rgb = self._camera.capture_array()
            frame_bgr = self._cv2.cvtColor(frame_rgb, self._cv2.COLOR_RGB2BGR)

            results = self._deepface.represent(
                img_path=frame_bgr,
                model_name=MODEL_NAME,
                enforce_detection=True,
                detector_backend='opencv',
            )
        except Exception:
            return []

        detections = []
        for r in results:
            embedding = np.array(r['embedding'])
            name, distance = self._best_match(embedding)
            if name:
                person = self._db[name]
                # Convert euclidean distance to 0–1 confidence
                confidence = round(max(0.0, 1.0 - distance / THRESHOLD), 3)
                detections.append({
                    'name':         name,
                    'confidence':   confidence,
                    'relationship': person['relationship'],
                })
        return detections

    # ── Matching ──────────────────────────────────────────────────────────────

    def _best_match(self, embedding: np.ndarray) -> tuple:
        best_name, best_dist = None, float('inf')
        for name, data in self._db.items():
            for stored in data['embeddings']:
                dist = float(np.linalg.norm(embedding - stored))
                if dist < best_dist:
                    best_dist = dist
                    best_name = name
        if best_dist < THRESHOLD:
            return best_name, best_dist
        return None, best_dist
