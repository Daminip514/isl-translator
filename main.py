import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pyttsx3
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Dummy model for gesture recognition (replace with your trained LSTM model)
# For demo, maps landmarks to gesture labels
gesture_labels = {0: "Hello", 1: "Thank You", 2: "Please"}  # Example labels
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, input_shape=(None, 63)),  # 21 landmarks * 3 (x, y, z)
    tf.keras.layers.Dense(len(gesture_labels), activation="softmax")
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

# Initialize text-to-speech
engine = pyttsx3.init()

# Tkinter UI
root = tk.Tk()
root.title("ISL Translator")
label = Label(root)
label.pack()

def process_frame():
    ret, frame = cap.read()
    if not ret:
        return
    frame = cv2.flip(frame, 1)  # Mirror frame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame_rgb, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            # Extract landmarks
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            landmarks = np.array([landmarks]).reshape(1, 1, -1)  # Shape for LSTM
            prediction = model.predict(landmarks)
            gesture_id = np.argmax(prediction)
            gesture_text = gesture_labels.get(gesture_id, "Unknown")
            engine.say(gesture_text)
            engine.runAndWait()
            cv2.putText(frame_rgb, gesture_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Update UI
    img = Image.fromarray(frame_rgb)
    imgtk = ImageTk.PhotoImage(image=img)
    label.imgtk = imgtk
    label.configure(image=imgtk)
    root.after(10, process_frame)

# Start webcam
cap = cv2.VideoCapture(0)
root.after(10, process_frame)
root.mainloop()
cap.release()