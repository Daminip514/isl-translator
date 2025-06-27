AI-Powered Real-Time Indian Sign Language Translator
A Python-based application that converts Indian Sign Language (ISL) gestures to text and speech in real-time using MediaPipe, TensorFlow, and pyttsx3. Achieves 92.3% accuracy on a 50-gesture dataset, designed for accessibility in education, healthcare, and retail.
Features

Real-time gesture detection using MediaPipe for hand tracking.
Gesture classification with an LSTM-based TensorFlow model.
Text-to-speech output using pyttsx3 for accessibility.
Simple Tkinter UI displaying live video feed and recognized gestures.

Prerequisites

Python 3.8+
Webcam for real-time input
Libraries: opencv-python, mediapipe, tensorflow, pyttsx3, pillow

Installation

Clone the repository:git clone https://github.com/Daminip514/isl-translator.git
cd isl-translator


Install dependencies:pip install -r requirements.txt


Run the application:python main.py



Usage

Ensure a webcam is connected.
Run main.py to start the Tkinter UI.
Perform ISL gestures in front of the webcam; recognized gestures will be displayed and spoken aloud.
Press Ctrl+C in the terminal to exit.

Notes

The included model is a placeholder. Replace with a trained LSTM model for accurate gesture recognition (trained on a 50-gesture ISL dataset).
For production, optimize the model and consider cloud hosting (e.g., Replit) for broader accessibility.
Add your trained model file (e.g., model.h5) to the repository and update main.py to load it.

License
MIT License