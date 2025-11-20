#!/usr/bin/env python3
"""
ai_glasses_onnx.py
YOLOv5n ONNX + ONNXRuntime object detection on Raspberry Pi (Picamera2).
Uses Vosk for offline voice commands and pico2wave + aplay for offline TTS.
Place yolov5n.onnx next to this script.
"""

import os
import time
import numpy as np
import cv2
from picamera2 import Picamera2
import onnxruntime as ort
import sounddevice as sd
import json
from vosk import Model as VoskModel, KaldiRecognizer
import subprocess
import datetime
from queue import Queue
import threading

# ---------- Config ----------
MODEL_PATH = "yolov5n.onnx"
CAM_RES = (320, 240)
CAM_FPS = 15
CONF_THRESHOLD = 0.3
VOSK_MODEL_PATH = os.path.expanduser("~/vosk-model-small-en-us-0.15")
# ----------------------------------------------------------------------

def tts_pico(text, wav="temp_tts.wav"):
    try:
        subprocess.run(["pico2wave", "-w", wav, text], check=True)
        subprocess.run(["aplay", wav], check=True)
        try:
            os.remove(wav)
        except:
            pass
    except Exception as e:
        print("TTS error:", e)

# non-blocking TTS worker
speech_q = Queue()
def speech_worker():
    while True:
        txt = speech_q.get()
        if txt is None:
            break
        tts_pico(txt)
t_thread = threading.Thread(target=speech_worker, daemon=True)
t_thread.start()

def speak(text):
    speech_q.put(text)

# ---------- Vosk setup ----------
if os.path.exists(VOSK_MODEL_PATH):
    vosk_model = VoskModel(VOSK_MODEL_PATH)
else:
    print("Vosk model not found at", VOSK_MODEL_PATH)
    vosk_model = None

def listen_vosk(duration=1.5, sr=16000):
    if vosk_model is None:
        return None
    try:
        rec = KaldiRecognizer(vosk_model, sr)
        data = sd.rec(int(duration*sr), samplerate=sr, channels=1, dtype='int16')
        sd.wait()
        if rec.AcceptWaveform(data.tobytes()):
            res = json.loads(rec.Result())
            return res.get("text","").lower()
        else:
            res = json.loads(rec.PartialResult())
            return res.get("partial","").lower()
    except Exception as e:
        print("Vosk listen error:", e)
        return None

# ---------- Load ONNX model ----------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"ONNX model not found: {MODEL_PATH}")

providers = ['CPUExecutionProvider']
sess = ort.InferenceSession(MODEL_PATH, providers=providers)
input_name = sess.get_inputs()[0].name
print("ONNX model loaded.")

# Helper: preprocess for YOLOv5 ONNX (assuming NCHW, scale 0-1, letterbox)
def letterbox(img, new_shape=(320,320), color=(114,114,114)):
    shape = img.shape[:2]  # current shape [h, w]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw //= 2; dh //= 2
    resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = dh, dh
    left, right = dw, dw
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return padded, r, (left, top)

def xywh2xyxy(x):
    # x is [x_center, y_center, w, h]
    y = np.copy(x)
    y[0] = x[0] - x[2]/2
    y[1] = x[1] - x[3]/2
    y[2] = x[0] + x[2]/2
    y[3] = x[1] + x[3]/2
    return y

# Colors for boxes
BOX_COLOR = (0,255,0)

# ---------- Camera ----------
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": CAM_RES})
picam2.configure(config)
picam2.start()
time.sleep(0.1)
print("Camera started.")

# ---------- Main loop ----------
frame_count = 0
last_voice_check = time.time()
speak("Smart glasses starting with ONNX detection")

try:
    while True:
        frame = picam2.capture_array()  # RGB
        img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # prepare input
        img0 = img.copy()
        img_in, ratio, (pad_x, pad_y) = letterbox(img0, new_shape=(CAM_RES[1], CAM_RES[0]))
        img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB)
        img_in = img_in.astype(np.float32) / 255.0
        img_in = np.transpose(img_in, (2,0,1))[np.newaxis, ...]  # 1x3xHxW

        # run inference
        pred = sess.run(None, {input_name: img_in})[0]  # shape depends on model export
        # Typical YOLOv5 ONNX export returns (1, boxes, 85) or similar. We need to parse.
        # We'll handle common case: outputs shape (1, N, 85) -> xywh + conf + classes
        if pred is None:
            continue
        out = pred[0] if pred.shape[0] == 1 else pred  # (N,85)
        # Filter by confidence (objectness * class conf)
        detection_texts = []
        for det in out:
            # det: [x, y, w, h, conf, class0_conf, class1_conf, ...] or sometimes [x,y,w,h,conf,cls]
            conf_obj = float(det[4])
            if conf_obj < CONF_THRESHOLD:
                continue
            # find class with max score (if class scores present)
            if det.shape[0] > 6:
                class_scores = det[5:]
                cls_idx = int(np.argmax(class_scores))
                cls_conf = float(class_scores[cls_idx])
                score = conf_obj * cls_conf
            else:
                cls_idx = int(det[5]) if det.shape[0] > 5 else 0
                score = conf_obj
            if score < CONF_THRESHOLD:
                continue
            xywh = det[:4]
            # convert xywh back to original image scale
            x_center, y_center, w, h = xywh
            # scale back using ratio and padding
            x_center = (x_center - pad_x) / ratio
            y_center = (y_center - pad_y) / ratio
            w = w / ratio
            h = h / ratio
            xyxy = [x_center - w/2, y_center - h/2, x_center + w/2, y_center + h/2]
            x1, y1, x2, y2 = map(int, xyxy)
            # label (we'll use COCO names if available file not included — default to class index)
            label = str(cls_idx)
            # distance estimation (fast heuristic)
            box_w = x2 - x1
            distance = None
            if box_w > 0:
                distance = (0.5 * 615) / box_w
            text = f"{label} {distance:.1f}m" if distance else label
            detection_texts.append(text)
            # draw
            cv2.rectangle(img, (x1,y1), (x2,y2), BOX_COLOR, 2)
            cv2.putText(img, text, (x1, max(0,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, BOX_COLOR, 1)

        # Voice command check every ~1s
        if (time.time() - last_voice_check) > 1.0 and vosk_model is not None:
            last_voice_check = time.time()
            cmd = listen_vosk(duration=1.2)
            if cmd:
                print("[VOICE]", cmd)
                if "stop" in cmd or "exit" in cmd or "quit" in cmd:
                    speak("Shutting down smart glasses")
                    break
                if "what time" in cmd or "time is it" in cmd:
                    now = datetime.datetime.now().strftime("%I:%M %p")
                    speak(f"The time is {now}")
                if "describe" in cmd or "surroundings" in cmd:
                    if detection_texts:
                        speak("Surroundings: " + ", ".join(detection_texts[:6]))
                    else:
                        speak("No objects detected nearby")

        frame_count += 1

except KeyboardInterrupt:
    print("Interrupted by user")

finally:
    try:
        picam2.close()
    except:
        pass
    speech_q.put(None)
    t_thread.join(timeout=1)
    print("Exiting.")
