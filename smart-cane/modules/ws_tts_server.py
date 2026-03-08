"""
ws_tts_server.py — WebSocket server that pushes TTS announcement text to the mobile app.

The Pi calls `send_text()` whenever it has a phrase to announce.
The mobile app connects here and handles ElevenLabs TTS + audio playback.

Port: 5001 (configurable)
"""

import asyncio
import json
import logging
import time
from typing import Set

import websockets
from websockets.server import WebSocketServerProtocol

logger = logging.getLogger(__name__)


class WSTTSServer:
    def __init__(self, host: str = '0.0.0.0', port: int = 5001):
        self.host = host
        self.port = port
        self.clients: Set[WebSocketServerProtocol] = set()
        self._server = None
        self._loop: asyncio.AbstractEventLoop | None = None

    # ─── Lifecycle ──────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the WebSocket server. Call this inside an asyncio event loop."""
        self._server = await websockets.serve(
            self._handler,
            self.host,
            self.port,
            ping_interval=20,   # keep-alive ping every 20s
            ping_timeout=10,
        )
        self._loop = asyncio.get_event_loop()
        logger.info(f'[WS TTS] Listening on ws://{self.host}:{self.port}')

    async def stop(self) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()

    # ─── Connection handler ──────────────────────────────────────────────────────

    async def _handler(self, ws: WebSocketServerProtocol) -> None:
        self.clients.add(ws)
        logger.info(f'[WS TTS] Client connected: {ws.remote_address}  (total: {len(self.clients)})')
        try:
            await ws.wait_closed()
        finally:
            self.clients.discard(ws)
            logger.info(f'[WS TTS] Client disconnected  (remaining: {len(self.clients)})')

    # ─── Send API ────────────────────────────────────────────────────────────────

    async def send_text(self, text: str, event_type: str = 'announcement', **extra) -> None:
        """Broadcast a TTS text message to all connected mobile clients."""
        if not self.clients:
            logger.debug('[WS TTS] No clients connected — skipping TTS push')
            return

        message = json.dumps({
            'text': text,
            'type': event_type,
            'timestamp': time.time(),
            **extra,
        })

        # Send to all clients concurrently; ignore individual failures
        results = await asyncio.gather(
            *[client.send(message) for client in self.clients],
            return_exceptions=True,
        )

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f'[WS TTS] Failed to send to client {i}: {result}')

    def send_text_threadsafe(self, text: str, event_type: str = 'announcement', **extra) -> None:
        """
        Thread-safe wrapper for calling send_text() from a non-async thread.
        Use this from main.py's obstacle/camera threads.
        """
        if self._loop and not self._loop.is_closed():
            asyncio.run_coroutine_threadsafe(
                self.send_text(text, event_type, **extra),
                self._loop,
            )


# Singleton instance — imported by main.py and voice_alert.py
tts_server = WSTTSServer()
