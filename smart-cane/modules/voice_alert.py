"""
voice_alert.py — Generates announcement text via Gemini Flash and pushes it
to the mobile app via WebSocket. ElevenLabs TTS is handled by the mobile app.

NOTE: ElevenLabs API call and audio playback have been REMOVED from this module.
      The Pi now only generates text; the phone handles TTS + playback.
"""

import logging
import os
import time
from typing import Optional

from google import genai

logger = logging.getLogger(__name__)

# ─── Gemini setup ─────────────────────────────────────────────────────────────

GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')
GEMINI_MODEL   = 'gemini-2.0-flash'

if GEMINI_API_KEY:
    _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
else:
    _gemini_client = None
    logger.warning('[VoiceAlert] GEMINI_API_KEY not set — using fallback phrases')

# ─── Phrase cache ─────────────────────────────────────────────────────────────
# Cache generated phrases per person so we don't call Gemini on every detection
_phrase_cache: dict[str, tuple[str, float]] = {}  # name → (phrase, expiry_ts)
PHRASE_CACHE_TTL_S = 3600  # regenerate phrase after 1 hour


# ─── Public API ───────────────────────────────────────────────────────────────

def generate_announcement(
    name: str,
    relationship: str = '',
    confidence: float = 1.0,
    ws_server=None,  # WSTTSServer instance
) -> Optional[str]:
    """
    Generate a warm, context-aware announcement phrase for a detected person.
    Pushes the phrase to the mobile app via WebSocket if ws_server is provided.
    Returns the phrase string (or None on failure).
    """
    phrase = _get_or_generate_phrase(name, relationship, confidence)
    if not phrase:
        return None

    logger.info(f'[VoiceAlert] Announcing: "{phrase}"')

    # Push to mobile app via WebSocket — include name + confidence for UI update
    if ws_server is not None:
        ws_server.send_text_threadsafe(
            phrase,
            event_type='face_announcement',
            name=name,
            confidence=round(confidence, 3),
        )
    else:
        logger.warning('[VoiceAlert] No WebSocket server — announcement not delivered')

    return phrase


def generate_obstacle_alert(distance_cm: int, ws_server=None) -> Optional[str]:
    """
    Generate a brief obstacle warning for very close obstacles.
    Only called for danger-level obstacles (< 50cm).
    """
    if distance_cm < 20:
        phrase = 'Warning — obstacle immediately ahead, stop now.'
    elif distance_cm < 50:
        phrase = f'Caution — obstacle about {distance_cm} centimeters ahead.'
    else:
        return None  # not close enough to warrant a voice alert

    logger.info(f'[VoiceAlert] Obstacle alert: "{phrase}"')

    if ws_server is not None:
        ws_server.send_text_threadsafe(phrase, event_type='obstacle_alert')

    return phrase


# ─── Phrase generation ───────────────────────────────────────────────────────

def _get_or_generate_phrase(name: str, relationship: str, confidence: float) -> Optional[str]:
    """Return cached phrase if fresh, otherwise call Gemini."""
    cached = _phrase_cache.get(name)
    if cached and cached[1] > time.time():
        return cached[0]

    phrase = _call_gemini(name, relationship, confidence)
    if phrase:
        _phrase_cache[name] = (phrase, time.time() + PHRASE_CACHE_TTL_S)
    return phrase


def _call_gemini(name: str, relationship: str, confidence: float) -> Optional[str]:
    if not _gemini_client:
        return _fallback_phrase(name, relationship)

    prompt = (
        f"Generate a warm, natural-sounding announcement for a smart cane app. "
        f"The cane's user is elderly or visually impaired. "
        f"The cane just detected {name}"
        + (f", who is a {relationship}" if relationship else "")
        + f" (recognition confidence: {int(confidence * 100)}%). "
        f"Write ONE short sentence (under 15 words) that the app will speak aloud. "
        f"Be friendly and reassuring. Do not include any formatting or quotes."
    )

    try:
        response = _gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )
        phrase = response.text.strip().strip('"').strip("'")
        return phrase if phrase else _fallback_phrase(name, relationship)
    except Exception as e:
        logger.warning(f'[VoiceAlert] Gemini error: {e}')
        return _fallback_phrase(name, relationship)


def _fallback_phrase(name: str, relationship: str) -> str:
    if relationship:
        return f'{name}, your {relationship}, is nearby.'
    return f'{name} is approaching.'
