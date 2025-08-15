\# Finger-count

A real-time hand gesture recognition system to count fingers using Python, OpenCV, and MediaPipe.



Overview:

Fingercount captures live webcam input and uses computer vision techniques to detect hands and count the number of fingers shown. It combines the power of OpenCV for video processing and Google's MediaPipe for hand tracking.



\## Features



\- Real-time webcam integration

\- Hand tracking using MediaPipe

\- Accurate finger counting using landmarks

\- Lightweight and beginner-friendly



\## Requirements



Make sure Python 3 is installed, then install the required libraries:




pip install opencv-python mediapipe





How to Run:

Run the following command in your terminal:



python FingerML.py



Ensure your webcam is connected and active.






How It Works:


Captures video from webcam using OpenCV



Uses MediaPipe to detect hand landmarks



Logic is applied to detect which fingers are up based on joint positions



Displays the count in real time




