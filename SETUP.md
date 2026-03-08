# 🎙️ Face Recognition Greeter — Setup Guide
## Raspberry Pi 5 + Camera Module 3 + MAX98357A + Speaker

---

## 📦 Software Installation

```bash
# System packages
sudo apt update && sudo apt install -y \
    python3-picamera2 \
    python3-opencv \
    cmake \
    libatlas-base-dev \
    libjpeg-dev \
    mpg123 \
    python3-pip

# Python packages
pip install \
    mediapipe \
    face_recognition \
    google-generativeai \
    elevenlabs \
    numpy \
    opencv-python \
    --break-system-packages
```

---

## 🔑 API Keys

Get your keys and export them (add to ~/.bashrc for persistence):

```bash
export GEMINI_API_KEY="your_key_here"
export ELEVENLABS_API_KEY="your_key_here"
export ELEVENLABS_VOICE_ID="Rachel"   # or your custom voice ID
```

---

## 🔌 MAX98357A Wiring — Raspberry Pi 5 GPIO

```
MAX98357A Pin    →   Raspberry Pi 5 Pin (GPIO)
─────────────────────────────────────────────
VIN              →   Pin 2  (5V)
GND              →   Pin 6  (GND)
BCLK             →   Pin 12 (GPIO18 — PCM_CLK / I2S Clock)
LRC  (WSEL)      →   Pin 35 (GPIO19 — PCM_FS  / I2S Word Select)
DIN              →   Pin 40 (GPIO21 — PCM_DOUT / I2S Data)
GAIN (optional)  →   Leave floating for 9dB gain
                     Connect to GND for 12dB gain
                     Connect to VIN for 15dB gain (loud!)
SD (Shutdown)    →   Leave unconnected or tie to 3.3V to keep on

Speaker wires    →   + and – terminals on MAX98357A output
                     (4Ω 3W mini speaker)
```

### Visual GPIO Diagram
```
Pi 5 Header (looking at board, pins 1-40 left→right, top→bottom)

 [1  3.3V] [2  5V  ] ← VIN here
 [3  SDA ] [4  5V  ]
 [5  SCL ] [6  GND ] ← GND here
 [7  GP4 ] [8  TX  ]
 [9  GND ] [10 RX  ]
 [11 GP17] [12 GP18] ← BCLK here
 [13 GP27] [14 GND ]
 [15 GP22] [16 GP23]
 [17 3.3V] [18 GP24]
 [19 MOSI] [20 GND ]
 [21 MISO] [22 GP25]
 [23 SCLK] [24 CE0 ]
 [25 GND ] [26 CE1 ]
 [27 SDA1] [28 SCL1]
 [29 GP5 ] [30 GND ]
 [31 GP6 ] [32 GP12]
 [33 GP13] [34 GND ]
 [35 GP19] [36 GP16] ← LRC here (pin 35)
 [37 GP26] [38 GP20]
 [39 GND ] [40 GP21] ← DIN here (pin 40)
```

---

## ⚙️ Enable I2S on Raspberry Pi 5

**Step 1: Edit /boot/firmware/config.txt**
```bash
sudo nano /boot/firmware/config.txt
```

Add these lines at the bottom:
```
# MAX98357A I2S DAC
dtoverlay=hifiberry-dac
```

**Step 2: Reboot**
```bash
sudo reboot
```

**Step 3: Verify I2S device appears**
```bash
aplay -l
# You should see a card like: card 0: sndrpihifiberry [snd_rpi_hifiberry_dac]
```

**Step 4: Set as default audio output**
```bash
sudo nano /etc/asound.conf
```
Paste:
```
pcm.!default {
    type hw
    card 0
}
ctl.!default {
    type hw
    card 0
}
```

**Step 5: Test audio**
```bash
mpg123 /usr/share/sounds/alsa/Front_Center.wav
# or download a test MP3 and play it
```

---

## 📷 Enable Camera Module 3

```bash
sudo nano /boot/firmware/config.txt
```
Ensure these are present:
```
camera_auto_detect=1
dtoverlay=camera-mux-2port  # only if using multiplexer
```

Test camera:
```bash
rpicam-still -o test.jpg
```

---

## 🚀 Running the System

### Step 1 — Capture reference photos for each person
```bash
python3 01_capture_faces.py --name "Ahmad" --photos 5
python3 01_capture_faces.py --name "Sara"  --photos 5
# Repeat for up to 5 people
```
This creates:
```
known_faces/
  Ahmad/
    Ahmad_001.jpg
    Ahmad_002.jpg
    ...
  Sara/
    Sara_001.jpg
    ...
```

### Step 2 — Start the recognition system
```bash
python3 02_face_recognition.py
```

### Run on boot (optional systemd service)
```bash
sudo nano /etc/systemd/system/face-greeter.service
```
```ini
[Unit]
Description=Face Recognition Greeter
After=network.target

[Service]
ExecStart=/usr/bin/python3 /home/pi/02_face_recognition.py
WorkingDirectory=/home/pi
Environment=GEMINI_API_KEY=your_key
Environment=ELEVENLABS_API_KEY=your_key
Restart=always
User=pi

[Install]
WantedBy=multi-user.target
```
```bash
sudo systemctl enable face-greeter
sudo systemctl start face-greeter
```

---

## 📋 Detection Log

Every recognized face is logged to `face_log.csv`:
```
timestamp,           name,  confidence, greeting
2025-01-15T08:32:11, Ahmad, 0.847,      "Ahmad is here! Good morning."
2025-01-15T09:14:05, Sara,  0.912,      "Hey Sara, welcome back!"
```

---

## 🔧 Tuning

| Setting | File | Default | Notes |
|---|---|---|---|
| Confidence threshold | `02_face_recognition.py` CONFIG | 0.70 | Lower = stricter |
| Cooldown between greetings | CONFIG | 30s | Increase to reduce repetition |
| Scan frequency | CONFIG | Every 5 frames | Lower number = more CPU |
| Gain on MAX98357A | Hardware | Float (9dB) | Wire GAIN to GND for louder |

---

## 🛠️ Troubleshooting

| Problem | Fix |
|---|---|
| No audio | Run `aplay -l` to confirm I2S device; check `/etc/asound.conf` |
| `face_recognition` install fails | `sudo apt install cmake libatlas-base-dev` first |
| Camera not found | `rpicam-still -o test.jpg` to verify; check `config.txt` |
| Gemini 400 error | Check API key; ensure billing enabled on Google Cloud |
| ElevenLabs error | Check API key; verify voice ID exists in your account |
| Face not recognized | Capture more reference photos; try different lighting |
