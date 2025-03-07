# Sign Language Translator

A real-time sign language translator that can recognize hand gestures using your webcam.

## Features

- Real-time hand gesture recognition
- Support for 10 common signs:
  - HELLO
  - YES
  - NO
  - THANK YOU
  - GOOD MORNING
  - GOOD NIGHT
  - PLEASE
  - SORRY
  - WELCOME
  - BYE
- Confidence score display
- Live camera feed
- Easy-to-use interface

## How to Use

1. Open the application in your web browser
2. Click "Start Camera" to begin
3. Show your hand clearly in front of the camera
4. Make one of the supported signs
5. Hold the sign steady for a moment to get a prediction
6. Click "Stop Camera" when you're done

## Technical Details

- Uses MediaPipe for hand tracking
- TensorFlow.js for sign recognition
- Real-time processing at 15 FPS
- Confidence threshold for accurate predictions

## API Endpoints

- `POST /process_frame`: Process a video frame and return predictions
- `GET /health`: Check API health status

## Live Demo

Visit the live demo at: [Your GitHub Pages URL] 