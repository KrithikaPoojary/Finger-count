# Finger-Count

A real-time hand gesture recognition system to count fingers using Python, OpenCV, and MediaPipe.

## Features
- Real-time webcam integration
- Hand tracking using MediaPipe
- Finger counting using hand landmarks

## Technologies Used
- Python
- OpenCV
- MediaPipe

## Installation

Install the required libraries:

```bash
pip install opencv-python mediapipe
```

## How to Run

Make sure your webcam is connected and run:

```bash
python FingerML.py
```

## How It Works
1. Captures live video using OpenCV.
2. Detects hand landmarks using MediaPipe.
3. Identifies raised fingers based on landmark positions.
4. Displays the finger count in real time.
