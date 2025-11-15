import cv2
import torch
from gtts import gTTS
import speech_recognition as sr
import numpy as np
import time
import threading
from queue import Queue
import datetime
import os
from picamera import PiCamera
from picamera.array import PiRGBArray
import pygame  # For playing audio on Pi
# import googlemaps  # For Google Maps integration

# ------------------ Configuration ------------------
# For Raspberry Pi 4: Use Pi Camera Module
CAMERA_RESOLUTION = (320, 240)  # Lower resolution for Pi performance
CAMERA_FRAMERATE = 15  # Lower FPS for Pi
MICROPHONE_DEVICE_INDEX = 0  # Set to 0 for Pi USB mic
VOICE_COMMAND_INTERVAL = 30  # Check voice commands every 30 frames (adjusted for Pi)
DETECTION_CONFIDENCE_THRESHOLD = 0.5  # Only process high-confidence detections
# GOOGLE_MAPS_API_KEY = "YOUR_GOOGLE_MAPS_API_KEY"  # Replace with your actual API key
# CURRENT_LOCATION = "Your Current Location"  # Set to actual location or use GPS if available

# ------------------ 1. Load YOLOv5 model ------------------
print("Loading YOLOv5 model...")
try:
    model = torch.hub.load('ultralytics/yolov5', 'yolov5n')  # Use nano model for Pi performance
    model.conf = DETECTION_CONFIDENCE_THRESHOLD  # Set confidence threshold
    model.to('cpu')  # Force CPU inference for Pi
    print("YOLOv5 model loaded successfully!")
except Exception as e:
    print(f"Error loading YOLOv5 model: {e}")
    exit(1)

# ------------------ 2. Optimized Text-to-Speech Engine ------------------
# Initialize pygame for audio playback on Pi
pygame.mixer.init()

# Use threading for non-blocking speech
speech_queue = Queue()
speech_thread = None

def speak_worker():
    """Background thread for speech processing using gTTS"""
    while True:
        text = speech_queue.get()
        if text is None:
            break
        try:
            tts = gTTS(text=text, lang='en', slow=False)
            tts.save("temp_audio.mp3")
            pygame.mixer.music.load("temp_audio.mp3")
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            os.remove("temp_audio.mp3")
        except Exception as e:
            print(f"TTS Error: {e}")

def speak(text):
    """Non-blocking speech function"""
    global speech_thread
    if speech_thread is None or not speech_thread.is_alive():
        speech_thread = threading.Thread(target=speak_worker, daemon=True)
        speech_thread.start()
    speech_queue.put(text)

# ------------------ 3. Optimized Speech Recognition Setup ------------------
recognizer = sr.Recognizer()
microphone = sr.Microphone(device_index=MICROPHONE_DEVICE_INDEX)

# Pre-adjust microphone once
try:
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.5)  # Faster adjustment
    speak("Microphone calibrated")
except Exception as e:
    print(f"Microphone setup error: {e}")

def listen_for_command(timeout=2):  # Reduced timeout for faster response
    """Fast voice command listening"""
    try:
        with microphone as source:
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=2)
            command = recognizer.recognize_google(audio).lower()
            print(f"[Voice Command]: {command}")
            return command
    except (sr.WaitTimeoutError, sr.UnknownValueError, sr.RequestError):
        return None

# ------------------ 4. Optimized Distance Estimation ------------------
def estimate_distance(box_width, focal_length=615, real_width=0.5):
    """Fast distance calculation with caching"""
    try:
        box_width = float(box_width)
        return (real_width * focal_length) / box_width if box_width > 0 else None
    except (ValueError, TypeError):
        return None

# ------------------ 5. Google Maps Navigation ------------------
# def get_directions(destination):
#     """Get directions from current location to destination using Google Maps API"""
#     try:
#         gmaps = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)
#         directions_result = gmaps.directions(CURRENT_LOCATION, destination, mode="walking")
#         if directions_result:
#             steps = directions_result[0]['legs'][0]['steps']
#             return steps
#         else:
#             return None
#     except Exception as e:
#         print(f"Google Maps Error: {e}")
#         return None

# def start_navigation(destination):
#     """Start turn-by-turn navigation"""
#     speak(f"Starting navigation to {destination}")
#     steps = get_directions(destination)
#     if steps:
#         for step in steps:
#             instruction = step['html_instructions'].replace('<b>', '').replace('</b>', '').replace('<div style="font-size:0.9em">', '').replace('</div>', '')
#             distance = step['distance']['text']
#             speak(f"{instruction}. Distance: {distance}")
#             time.sleep(2)  # Brief pause between instructions
#         speak("You have arrived at your destination")
#     else:
#         speak("Unable to find directions to that location")

# ------------------ 5. Main Smart Glasses Loop ------------------
# Initialize Pi Camera
camera = PiCamera()
camera.resolution = CAMERA_RESOLUTION
camera.framerate = CAMERA_FRAMERATE
raw_capture = PiRGBArray(camera, size=CAMERA_RESOLUTION)

# Allow camera to warm up
time.sleep(0.1)
speak("Smart glasses starting with optimized detection...")

# Performance tracking
last_detection_time = 0
detection_interval = 0.2  # Process every 200ms for Pi performance

# Global for surroundings command
detection_texts = []

frame_count = 0
for frame in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):
    frame = frame.array

    # Only process detections at intervals for better performance
    current_time = time.time()
    if current_time - last_detection_time >= detection_interval:
        results = model(frame)
        detections = results.xyxy[0]
        
        # Batch process detections
        detection_texts = []
        for *xyxy, conf, cls in detections:
            if conf < DETECTION_CONFIDENCE_THRESHOLD:
                continue
                
            label = model.names[int(cls)]
            x1, y1, x2, y2 = map(int, xyxy)
            box_width = x2 - x1
            
            distance = estimate_distance(box_width)
            if distance and distance < 10:
                detection_texts.append(f"{label} at {distance:.1f}m")
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {distance:.1f}m" if distance else label,
                        (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        

        
        last_detection_time = current_time

    # Fast voice command checking
    if frame_count % VOICE_COMMAND_INTERVAL == 0:
        command = listen_for_command(timeout=1)
        if command:
            if any(word in command for word in ["stop", "quit", "exit"]):
                speak("Shutting down smart glasses.")
                break
            elif "help" in command:
                speak("Say stop to quit, find route to place, what time is it, describe surroundings, or identify objects")
            elif "find route to" in command:
                place = command.replace("find route to", "").strip()
                if place:
                    start_navigation(place)
                else:
                    speak("Please specify a place for the route")
            elif "what time is it" in command:
                current_time = datetime.datetime.now().strftime("%I:%M %p")
                speak(f"The current time is {current_time}")
            elif "describe surroundings" in command:
                if detection_texts:
                    speak("Surroundings: " + ", ".join(detection_texts))
                else:
                    speak("No objects detected nearby")
            elif "identify objects" in command:
                if detection_texts:
                    speak("Identified objects: " + ", ".join(detection_texts))
                else:
                    speak("No objects detected nearby")
    
    # Clear the stream for next frame
    raw_capture.truncate(0)

    # Check for quit key (since no GUI on Pi, use keyboard interrupt or remove)
    try:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            speak("Shutting down smart glasses.")
            break
    except:
        pass  # Ignore on headless Pi

    frame_count += 1

# Cleanup
camera.close()
cv2.destroyAllWindows()
if speech_thread and speech_thread.is_alive():
    speech_queue.put(None)
    speech_thread.join(timeout=1)
