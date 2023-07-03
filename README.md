# Sign-Language-and-Emotion-Detection

## Description

This repository contains a comprehensive project on sign language and emotion detection using deep learning techniques. The project includes two trained models: one for sign language detection and another for emotion detection.

## Sign Language Detection

The sign language detection model is designed to identify American character sign language from an input image and provide corresponding character predictions. It leverages the power of MediaPipe, a popular library for hand tracking and landmark detection, to accurately detect and track hand movements. The hand landmarks extracted from the detected hand region are then used as input for a Random Forest classifier, which makes the final sign language prediction.<br>
Refer to directory - `Sign_language_Training` for more insight.

## Emotion Detection

The emotion detection model utilizes a Convolutional Neural Network (CNN) architecture to classify facial expressions into several emotion categories. The model is trained on a diverse dataset of facial images representing emotions such as Anger, Disgust, Fear, Happy, Neutral, Sadness, and Surprise. Given an input image, the model extracts the face region using a Haar cascade classifier and predicts the corresponding emotion using the trained CNN model.<br>
Refer to directory - `Emotion_Training` for more insight.

## Flask App implementation

The repository also includes a Flask application that demonstrates the real-time functionality of both the sign language and emotion detection models. The sign language component allows users to interact with the system by providing a live video feed from their camera and receiving predictions for the hand signs they depict. The emotion detection component accepts user-provided images and provides predictions for the emotions expressed in those images.<br>
Refer to directory - `Flask_app` for more insight.
