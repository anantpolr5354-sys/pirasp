"""
Rewritten ai_glasses_direct.py for Raspberry Pi 4 (Bullseye/Bookworm) using:
 - Picamera2 (camera)
 - ultralytics YOLO (lightweight detection)
 - Vosk (offline speech recognition)
 - pico2wave + aplay (offline TTS)

Save this file as ai_glasses_pi_ready.py on the Pi and run with python3.

Dependencies (install before running):
# apt packages
sudo apt update && sudo apt install -y python3-picamera2 python3-opencv libatlas-base-dev \
    libsndfile1 ffmpeg libttspico-utils aplay wget build-essential

# python packages (inside venv recommended)
python3 -m pip install --upgrade pip
python3 -m pip install ultralytics vosk numpy pygame

# Vosk model (one-time):
# Download a small-en-us model (e.g. vosk-model-small-en-us-0.15) and unzip to /home/pi/vosk-model-small
# Example (run once):
# wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
# unzip vosk-model-small-en-us-0.15.zip -d /home/pi/

Notes:
 - Replace GOOGLE_MAPS_API_KEY with your key if you want navigation (requires internet).
 - This script uses offline TTS (pico2wave) and offline ASR (Vosk) for responsive performance.
 - ultralytics YOLO expects model weights (it will auto-download yolov8n by default on first run).

"""

import os
import time
import threading
import datetime
from queue import Queue

import numpy as np
import cv2

# Camera: Picamera2
from picamera2 import Picamera2

# Object detection: ultralytics YOLO
from ultralytics import YOLO

# Speech recognition: Vosk (offline)
from vosk import Model as VoskModel, KaldiRecognizer
import sounddevice as sd
import json

# TTS: pico2wave (offline) + aplay to play wav
import subprocess

# Google Maps (optional)
try:
    import googlemaps
except Exception:
    googlemaps = None

# ---------------- Configuration ----------------
CAMERA_RESOLUTION = (320, 240)
CAMERA_FRAMERATE = 15
VOICE_COMMAND_CHECK_SEC = 1.0  # interval to check for voice commands (seconds)
DETECTION_CONFIDENCE_THRESHOLD = 0.3
GOOGLE_MAPS_API_KEY = os.environ.get('GOOGLE_MAPS_API_KEY', '')  # set via env var if needed
CURRENT_LOCATION = "Your Current Location"  # placeholder
VOSK_MODEL_PATH = os.path.expanduser("~/vosk-model-small-en-us-0.15")  # adjust to where you unzipped

# ------------------ Utilities ------------------

def tts_pico(text: str, filename: str = "temp_tts.wav"):
    """Generate speech using pico2wave and play with aplay (blocking)."""
    try:
        # Create wave file
        cmd = ["pico2wave", "-w", filename, text]
        subprocess.run(cmd, check=True)
        # Play it
        subprocess.run(["aplay", filename], check=True)
        # Remove file
        try:
            os.remove(filename)
        except Exception:
            pass
    except Exception as e:
        print(f"TTS error: {e}")

# Non-blocking TTS queue + worker
speech_queue = Queue()

def speech_worker():
    while True:
        txt = speech_queue.get()
        if txt is None:
            break
        tts_pico(txt)

speech_thread = threading.Thread(target=speech_worker, daemon=True)
speech_thread.start()

def speak(text: str):
    speech_queue.put(text)

# ------------------ Vosk (offline ASR) ------------------
if not os.path.exists(VOSK_MODEL_PATH):
    print(f"Warning: Vosk model not found at {VOSK_MODEL_PATH}. Voice commands will be disabled until model is downloaded.")
    vosk_model = None
else:
    try:
        vosk_model = VoskModel(VOSK_MODEL_PATH)
        # Use 16kHz mono audio for Vosk
        rec = KaldiRecognizer(vosk_model, 16000)
    except Exception as e:
        print(f"Failed to load Vosk model: {e}")
        vosk_model = None
        rec = None

# Function to record a short clip and run recognition
def listen_offline(duration=2.0, samplerate=16000):
    if vosk_model is None:
        return None
    try:
        audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
        sd.wait()
        data = audio.tobytes()
        r = KaldiRecognizer(vosk_model, samplerate)
        if r.AcceptWaveform(data):
            res = json.loads(r.Result())
            text = res.get('text', '').lower()
            return text
        else:
            res = json.loads(r.PartialResult())
            return res.get('partial', '').lower()
    except Exception as e:
        print(f"Vosk listening error: {e}")
        return None

# ------------------ Camera Setup (Picamera2) ------------------
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": CAMERA_RESOLUTION})
picam2.configure(config)
picam2.start()
print("Camera started with Picamera2")

# ------------------ Load YOLO Model (ultralytics) ------------------
print("Loading YOLO model (yolov8n)...")
try:
    yolo = YOLO('yolov8n.pt')  # will download if not present
    print("YOLO model ready")
except Exception as e:
    print(f"Failed to load YOLO model: {e}")
    yolo = None

# ------------------ Google Maps client (optional) ------------------
if GOOGLE_MAPS_API_KEY and googlemaps is not None:
    gmaps = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)
else:
    gmaps = None

# ------------------ Helper: estimate distance (simple) ------------------
def estimate_distance(box_width_px, focal_length=615, real_width_m=0.5):
    try:
        if box_width_px <= 0:
            return None
        return (real_width_m * focal_length) / float(box_width_px)
    except Exception:
        return None

# ------------------ Main Loop ------------------

try:
    speak("Smart glasses starting")
    last_voice_check = time.time()

    detection_texts = []
    frame_count = 0

    while True:
        frame = picam2.capture_array()
        # frame is RGB; convert to BGR for OpenCV display/drawing
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Run detection occasionally to save CPU
        if yolo is not None and frame_count % 1 == 0:
            results = yolo(frame, imgsz=CAMERA_RESOLUTION, conf=DETECTION_CONFIDENCE_THRESHOLD, verbose=False)
            detection_texts = []
            # ultralytics returns results list
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    conf = float(box.conf[0]) if box.conf is not None else 0.0
                    cls_idx = int(box.cls[0]) if box.cls is not None else None
                    if conf < DETECTION_CONFIDENCE_THRESHOLD:
                        continue
                    xyxy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(int, xyxy)
                    label = yolo.model.names[cls_idx] if cls_idx is not None and cls_idx in yolo.model.names else str(cls_idx)
                    box_w = x2 - x1
                    distance = estimate_distance(box_w)
                    if distance and distance < 10:
                        detection_texts.append(f"{label} at {distance:.1f}m")
                    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(frame_bgr, f"{label} {distance:.1f}m" if distance else label,
                                (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)

        # Periodically check for voice commands (non-blocking)
        if vosk_model is not None and (time.time() - last_voice_check) >= VOICE_COMMAND_CHECK_SEC:
            last_voice_check = time.time()
            cmd = listen_offline(duration=1.5)
            if cmd:
                print(f"[Voice] {cmd}")
                if any(w in cmd for w in ["stop", "quit", "exit"]):
                    speak("Shutting down smart glasses")
                    break
                if "what time" in cmd or "time is it" in cmd:
                    now = datetime.datetime.now().strftime("%I:%M %p")
                    speak(f"The time is {now}")
                elif "describe surroundings" in cmd or "describe" in cmd:
                    if detection_texts:
                        speak("Surroundings: " + ", ".join(detection_texts[:6]))
                    else:
                        speak("No objects detected nearby")
                elif "identify" in cmd or "what is" in cmd:
                    if detection_texts:
                        speak("Identified: " + ", ".join(detection_texts[:6]))
                    else:
                        speak("No objects detected nearby")
                elif "find route to" in cmd and gmaps is not None:
                    dest = cmd.split("find route to")[-1].strip()
                    if dest:
                        speak(f"Searching route to {dest}")
                        try:
                            directions = gmaps.directions(CURRENT_LOCATION, dest, mode="walking")
                            if directions:
                                steps = directions[0]['legs'][0]['steps']
                                for step in steps:
                                    instr = step['html_instructions']
                                    # strip HTML tags naively
                                    instr = instr.replace('<b>', '').replace('</b>', '').replace('<div style="font-size:0.9em">', '').replace('</div>', '')
                                    speak(instr)
                                    time.sleep(1)
                                speak("You have arrived")
                            else:
                                speak("No route found")
                        except Exception as e:
                            print(f"GMAP error: {e}")
                            speak("Error getting directions")
                    else:
                        speak("Please say the place name")

        # Optional: show frame on attached display (if available)
        # Convert back to RGB for proper display if needed
        # cv2.imshow('Frame', frame_bgr)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

        frame_count += 1

except KeyboardInterrupt:
    print("Interrupted by user")

finally:
    try:
        picam2.close()
    except Exception:
        pass
    # stop speech thread
    speech_queue.put(None)
    speech_thread.join(timeout=1)
    print("Cleaned up and exiting")
