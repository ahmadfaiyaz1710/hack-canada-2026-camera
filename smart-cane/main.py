"""
main.py — Smart Cane orchestrator (Raspberry Pi 5)

Threads:
  1. ws_tts_thread    — WebSocket server (port 5001) — pushes TTS text to phone
  2. face_api_thread  — HTTP REST server (port 5000) — face enrollment API
  3. camera_thread    — DeepFace recognition + WebSocket TTS push
  4. ble_thread       — GATT server for live BLE data to companion app
  5. obstacle_thread  — ultrasonic sensors + buzzer (if hardware present)

Shared state dict is the single source of truth between threads.
"""

import asyncio
import logging
import os
import threading
import time

from modules.ws_tts_server import tts_server
from modules.voice_alert import generate_announcement, generate_obstacle_alert
import modules.face_api_server as face_api

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(name)s: %(message)s',
)
logger = logging.getLogger('main')

# ─── Shared state ─────────────────────────────────────────────────────────────

state = {
    'obstacle_distance_cm': None,   # int | None
    'obstacle_severity': 'unknown', # 'safe' | 'caution' | 'danger' | 'unknown'
    'last_face': None,              # {'name': str, 'confidence': float, 'timestamp': float}
    'gps': None,                    # {'lat': float, 'lng': float, 'accuracy': float}
    'battery_pct': 100,
    'sos_active': False,
    'running': True,
}
state_lock = threading.Lock()


# ─── WebSocket TTS thread ─────────────────────────────────────────────────────

def ws_tts_thread():
    """Run the WebSocket TTS server in its own asyncio event loop."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def run():
        await tts_server.start()
        # Keep running until main sets running=False
        while state['running']:
            await asyncio.sleep(1)
        await tts_server.stop()

    loop.run_until_complete(run())
    loop.close()


# ─── Obstacle thread ─────────────────────────────────────────────────────────

def obstacle_thread():
    """Read ultrasonic sensors, drive buzzer, push voice alert on danger."""
    try:
        from modules.obstacle_detector import ObstacleDetector
        detector = ObstacleDetector()
    except ImportError:
        logger.warning('[obstacle] obstacle_detector module not found — skipping')
        return

    last_alert_ts = 0.0
    ALERT_COOLDOWN_S = 10  # don't repeat obstacle voice alert within 10s

    while state['running']:
        distance_cm = detector.measure()

        with state_lock:
            state['obstacle_distance_cm'] = distance_cm
            if distance_cm is None:
                state['obstacle_severity'] = 'unknown'
            elif distance_cm < 50:
                state['obstacle_severity'] = 'danger'
            elif distance_cm < 200:
                state['obstacle_severity'] = 'caution'
            else:
                state['obstacle_severity'] = 'safe'

        # Voice alert for danger-level obstacles (throttled)
        if (distance_cm is not None
                and distance_cm < 50
                and time.time() - last_alert_ts > ALERT_COOLDOWN_S):
            generate_obstacle_alert(distance_cm, ws_server=tts_server)
            last_alert_ts = time.time()

        time.sleep(0.1)  # 10 Hz


# ─── Face API thread ──────────────────────────────────────────────────────────

def face_api_thread():
    """Run the Flask face enrollment HTTP server on port 5000."""
    face_api.run(host='0.0.0.0', port=5000)


# ─── Camera / face recognition thread ────────────────────────────────────────

def camera_thread():
    """Detect faces, generate announcements, push via WebSocket."""
    try:
        from modules.face_recognizer import FaceRecognizer
        recognizer = FaceRecognizer()
        # Wire face_api_server to reload DB when a new face is enrolled
        face_api.set_db_updated_callback(recognizer.reload_db)
    except ImportError:
        logger.warning('[camera] face_recognizer module not found — skipping')
        return

    last_announced: dict[str, float] = {}  # name → last announcement timestamp
    FACE_ANNOUNCE_COOLDOWN_S = 30  # don't re-announce same person within 30s

    while state['running']:
        detections = recognizer.detect()  # returns list of {'name', 'confidence', 'relationship'}

        for det in detections:
            name = det.get('name', 'Unknown')
            confidence = det.get('confidence', 0.0)
            relationship = det.get('relationship', '')

            # Skip if obstacle is too close (safety priority)
            with state_lock:
                obstacle_cm = state['obstacle_distance_cm']
            if obstacle_cm is not None and obstacle_cm < 100:
                logger.debug(f'[camera] Suppressing face announcement — obstacle at {obstacle_cm}cm')
                continue

            # Throttle per-person announcements
            now = time.time()
            if now - last_announced.get(name, 0) < FACE_ANNOUNCE_COOLDOWN_S:
                continue

            phrase = generate_announcement(
                name=name,
                relationship=relationship,
                confidence=confidence,
                ws_server=tts_server,
            )

            if phrase:
                last_announced[name] = now
                with state_lock:
                    state['last_face'] = {
                        'name': name,
                        'confidence': confidence,
                        'timestamp': now,
                    }

        time.sleep(0.5)  # 2 Hz face detection polling


# ─── BLE thread ──────────────────────────────────────────────────────────────

def ble_thread():
    """Run GATT BLE server to push live data to mobile app."""
    try:
        from modules.ble_server import BLEServer
        server = BLEServer(state, state_lock)
        server.run()  # blocks until state['running'] = False
    except ImportError:
        logger.warning('[ble] ble_server module not found — skipping')


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    logger.info('Smart Cane starting up...')

    threads = [
        threading.Thread(target=ws_tts_thread,  name='ws_tts',   daemon=True),
        threading.Thread(target=face_api_thread, name='face_api', daemon=True),
        threading.Thread(target=camera_thread,   name='camera',   daemon=True),
        threading.Thread(target=ble_thread,      name='ble',      daemon=True),
        threading.Thread(target=obstacle_thread, name='obstacle', daemon=True),
    ]

    for t in threads:
        t.start()
        logger.info(f'Started thread: {t.name}')

    # Give WS server a moment to bind before other threads try to use it
    time.sleep(1)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info('Shutting down...')
        state['running'] = False
        for t in threads:
            t.join(timeout=3)
        logger.info('Goodbye.')


if __name__ == '__main__':
    main()
