from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import base64
import io
import time
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Load the model
try:
    model = tf.keras.models.load_model('ai/models/hand_gesture_model.keras')
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    # Create a new model with the correct configuration
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(42,)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    logger.info("Created new model with correct configuration")

# Define the signs
SIGNS = ['HELLO', 'YES', 'NO', 'THANK YOU', 'GOOD MORNING', 'GOOD NIGHT', 'PLEASE', 'SORRY', 'WELCOME', 'BYE']

# Initialize performance tracking variables
last_process_time = time.time()
process_interval = 1/15  # Process every 15th frame (15 FPS)
frame_count = 0
prediction_buffer = []
buffer_size = 5
min_confidence_threshold = 0.6
high_confidence_threshold = 0.85

def extract_hand_features(hand_landmarks):
    """Extract normalized hand features from landmarks"""
    if not hand_landmarks:
        return None
    
    # Get the hand center (middle finger base)
    hand_center = np.array([
        hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_BASE].x,
        hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_BASE].y
    ])
    
    # Extract features relative to hand center
    features = []
    for landmark in hand_landmarks.landmark:
        # Normalize coordinates relative to hand center
        x = landmark.x - hand_center[0]
        y = landmark.y - hand_center[1]
        features.extend([x, y])
    
    return np.array(features)

@app.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        # Get the frame data from the request
        frame_data = request.json.get('frame')
        if not frame_data:
            return jsonify({'error': 'No frame data provided'}), 400

        # Decode base64 image
        frame_bytes = base64.b64decode(frame_data.split(',')[1])
        frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
        
        # Skip frames to maintain target FPS
        current_time = time.time()
        if current_time - last_process_time < process_interval:
            return jsonify({'status': 'skipped'})
        
        # Process frame with MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            features = extract_hand_features(hand_landmarks)
            
            if features is not None:
                # Make prediction
                prediction = model.predict(features.reshape(1, -1), verbose=0)
                predicted_sign = SIGNS[np.argmax(prediction)]
                confidence = float(np.max(prediction))
                
                # Update prediction buffer
                prediction_buffer.append((predicted_sign, confidence))
                if len(prediction_buffer) > buffer_size:
                    prediction_buffer.pop(0)
                
                # Calculate average confidence and most common sign
                if len(prediction_buffer) >= 3:
                    avg_confidence = sum(c for _, c in prediction_buffer) / len(prediction_buffer)
                    signs = [s for s, _ in prediction_buffer]
                    most_common_sign = max(set(signs), key=signs.count)
                    sign_frequency = signs.count(most_common_sign) / len(signs)
                    
                    # Only show prediction if confidence and stability are high enough
                    if avg_confidence >= high_confidence_threshold and sign_frequency >= 0.8:
                        return jsonify({
                            'status': 'success',
                            'prediction': most_common_sign,
                            'confidence': avg_confidence,
                            'message': 'High confidence prediction'
                        })
                    elif avg_confidence >= min_confidence_threshold:
                        return jsonify({
                            'status': 'success',
                            'prediction': most_common_sign,
                            'confidence': avg_confidence,
                            'message': 'Low confidence prediction'
                        })
                    else:
                        return jsonify({
                            'status': 'success',
                            'message': 'Make a clearer sign'
                        })
        
        last_process_time = current_time
        return jsonify({'status': 'no_hand'})
        
    except Exception as e:
        logger.error(f"Error processing frame: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    # Warm up the model
    dummy_input = np.random.rand(1, 42)
    for _ in range(3):
        model.predict(dummy_input, verbose=0)
    
    logger.info("Model warmed up")
    logger.info("Available signs: " + ", ".join(SIGNS))
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000))) 