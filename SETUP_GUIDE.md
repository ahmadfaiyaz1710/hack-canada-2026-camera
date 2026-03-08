# Face Recognition Greeter — Setup Guide

## System Overview

```
┌─────────────────────────────────┐       Bluetooth RFCOMM
│   Raspberry Pi 5                │ ─────────────────────────► Android Phone
│                                 │                              (Face Greeter App)
│  ┌──────────┐  ┌─────────────┐  │
│  │ Camera   │  │ Program 2   │  │
│  │ Module 3 │→ │ (recognize) │──┼─── Gemini API (internet)
│  └──────────┘  └─────────────┘  │
│                                 │
│  Program 1  (enroll faces)      │
└─────────────────────────────────┘
```

---

## Wiring Schematic — Camera Module 3

The Camera Module 3 connects via the **CSI (Camera Serial Interface)** ribbon cable.
No extra wiring is needed — it's a direct plug-in connector.

```
Raspberry Pi 5 Board (top view, camera connector)
─────────────────────────────────────────────────
                        ┌──────────┐
                        │  Pi 5    │
                        │          │
  CAM0 connector ──────►│ [==CSI=] │◄────── CAM1 connector
  (15-pin FPC)          │          │         (22-pin FPC)
                        └──────────┘

Camera Module 3 ships with a 200mm FFC/FPC ribbon cable.

IMPORTANT: Pi 5 uses a DIFFERENT (smaller) connector than Pi 4.
  • Pi 5 CAM0 port = 15-pin, 1mm pitch FPC
  • Pi 5 CAM1 port = 22-pin, 0.5mm pitch FPC  ← Camera Mod 3 default cable fits here

Steps:
  1. Power OFF the Pi.
  2. Lift the locking tab on the CAM1 connector (white bar, pull upward).
  3. Slide the ribbon cable in with the BLUE side facing the USB ports.
  4. Press the locking tab back down firmly.
  5. Power ON.

If your kit includes a SHORTER cable (15-pin) for the CAM0 port, the process
is identical but the connector is smaller — refer to the official Raspberry Pi
Camera Module 3 documentation for the adapter board if needed.
```

### Physical Pinout (for reference — no extra wiring needed)
```
CSI Connector carries:
  • MIPI CSI-2 data lanes (D0+/D0-, D1+/D1-)
  • Clock lane (CLK+/CLK-)
  • I2C (SDA/SCL) for camera control
  • 3.3V power
  • GND
All managed automatically by picamera2 / libcamera.
```

---

## Software Installation on Raspberry Pi 5

### 1. Update OS & enable camera
```bash
sudo apt update && sudo apt upgrade -y
# Enable camera in raspi-config if not already:
sudo raspi-config
# → Interface Options → Camera → Enable
# (On Pi 5 running Bookworm, libcamera is enabled by default)
```

### 2. Install system dependencies
```bash
sudo apt install -y \
    python3-pip python3-opencv \
    libatlas-base-dev libhdf5-dev \
    bluetooth bluez python3-dev libbluetooth-dev
```

### 3. Install Python packages
```bash
pip3 install --break-system-packages \
    picamera2 \
    deepface \
    tf-keras \
    opencv-python-headless \
    google-generativeai \
    pybluez2 \
    numpy
```

> **Note:** DeepFace will download the ArcFace model weights (~250 MB) on first run.
> Run `python3 -c "from deepface import DeepFace; DeepFace.build_model('ArcFace')"` once
> while connected to Wi-Fi to pre-download.

### 4. Set environment variables
Add to `~/.bashrc`:
```bash
export GEMINI_API_KEY="your_gemini_api_key_here"
export BT_DEVICE_ADDR="AA:BB:CC:DD:EE:FF"   # your Android's Bluetooth MAC
export BT_PORT=1
```
Then: `source ~/.bashrc`

### 5. Get your Android Bluetooth MAC address
- Android: Settings → About Phone → Status → Bluetooth Address
  (or use: `bluetoothctl scan on` from the Pi after pairing)

### 6. Pair Pi ↔ Android
```bash
bluetoothctl
  power on
  agent on
  scan on
  # wait until your Android appears, note its address
  pair AA:BB:CC:DD:EE:FF
  trust AA:BB:CC:DD:EE:FF
  quit
```
Also pair from the Android side via Settings → Bluetooth.

---

## Usage

### Enroll a face (Program 1)
```bash
python3 program1_enroll_faces.py --name "Ahmad"
# Captures 8 samples by default. Press SPACE or wait for auto-capture.

# List enrolled people:
python3 program1_enroll_faces.py --list

# Delete someone:
python3 program1_enroll_faces.py --delete "Ahmad"
```

### Run the recognizer (Program 2)
```bash
python3 program2_recognize.py
```
- Opens a live preview window.
- When a known face is detected with similarity ≥ 0.70, Gemini generates a greeting.
- Greeting is sent over Bluetooth and spoken aloud on the Android app.

---

## Android App Setup

1. Open the project in **Android Studio** (min SDK 26).
2. Build & install on your Android device.
3. Pair with the Pi via Android Bluetooth Settings first.
4. Open the app → select "raspberrypi" (or whatever name your Pi shows) → tap **Connect**.
5. The app listens for incoming lines and displays + speaks them via TTS.

---

## Tuning Tips

| Parameter | File | Default | Effect |
|-----------|------|---------|--------|
| `THRESHOLD` | program2 | 0.70 | Higher = stricter (fewer false positives) |
| `COOLDOWN_SEC` | program2 | 10 | Seconds before same person triggers again |
| `DEFAULT_SAMPLES` | program1 | 8 | More samples = more robust matching |
| `detector_backend` | both | `"opencv"` | Swap to `"retinaface"` for better detection |

---

## Troubleshooting

**Camera not found:**
```bash
libcamera-hello   # should open a preview if camera is wired correctly
```

**Bluetooth errors:**
```bash
sudo systemctl status bluetooth
sudo systemctl restart bluetooth
```

**DeepFace model download fails:**
- Pre-download on a machine with better internet, copy `~/.deepface/` to the Pi.

**Low recognition accuracy:**
- Enroll more samples (try `--samples 15`).
- Use `detector_backend="retinaface"` in both scripts (slower but more accurate).
- Ensure consistent lighting during enrolment and recognition.

---

## Bill of Materials

| Item | Notes |
|------|-------|
| Raspberry Pi 5 (4GB or 8GB) | Main compute board |
| Raspberry Pi Camera Module 3 | Wide or standard; includes FPC cable |
| MicroSD card (32GB+, A2 class) | For OS |
| Pi 5 official power supply (27W USB-C) | Pi 5 needs more power than Pi 4 |
| Android phone (API 26+) | Any modern Android |
| Pi 5 case with camera mount (optional) | e.g. official Raspberry Pi case for Pi 5 |

No breadboard, no resistors, no GPIO wiring needed.
The Camera Module 3 is purely a ribbon-cable plug-in connection.
