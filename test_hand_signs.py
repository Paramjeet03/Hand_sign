import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import logging
import sys
import os
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Define available signs
SIGNS = ['HELLO', 'YES', 'NO', 'I LOVE YOU', 'GOOD', 'THANK YOU', 'Cute', "What", "Who"]

class HandSignDetector:
    def __init__(self):
        try:
            self.model = tf.keras.models.load_model('ai/models/hand_gesture_model.keras')
            self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            logging.info("Model loaded successfully")
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

    def extract_hand_features(self, hand_landmarks):
        try:
            # Normalize hand landmarks
            features = []
            for landmark in hand_landmarks.landmark:
                features.extend([landmark.x, landmark.y, landmark.z])
            features = np.array(features).reshape(1, -1)
            return features
        except Exception as e:
            logging.error(f"Error extracting features: {str(e)}")
            return None

    def predict(self, features):
        try:
            if features is None:
                return None, 0.0
            prediction = self.model.predict(features, verbose=0)
            predicted_class = np.argmax(prediction[0])
            confidence = prediction[0][predicted_class]
            return SIGNS[predicted_class], confidence
        except Exception as e:
            logging.error(f"Error during prediction: {str(e)}")
            return None, 0.0

def main():
    try:
        detector = HandSignDetector()
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            logging.error("Could not open webcam")
            return

        # Set video properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('hand_sign_detection_output.mp4', fourcc, 20.0, (640, 480))

        logging.info("Starting webcam feed. Recording will be saved to hand_sign_detection_output.mp4")
        logging.info("Press 'q' to quit")

        start_time = time.time()
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                logging.error("Failed to grab frame")
                break

            # Flip the frame horizontally for a later selfie-view display
            frame = cv2.flip(frame, 1)
            
            # Convert the BGR image to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame and detect hands
            results = detector.hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks
                    detector.mp_draw.draw_landmarks(
                        frame, hand_landmarks, detector.mp_hands.HAND_CONNECTIONS)
                    
                    # Extract features and predict
                    features = detector.extract_hand_features(hand_landmarks)
                    sign, confidence = detector.predict(features)
                    
                    if sign and confidence > 0.7:
                        # Add prediction text
                        text = f"{sign} ({confidence:.2f})"
                        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Add guide text
            guide_text = "Position your hand in the center of the frame"
            cv2.putText(frame, guide_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Write frame to output video
            out.write(frame)

            # Try to show the frame
            try:
                cv2.imshow('Hand Sign Detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except Exception as e:
                logging.warning(f"Could not display frame: {str(e)}")
                # Continue recording even if display fails

            frame_count += 1
            if frame_count % 30 == 0:
                elapsed_time = time.time() - start_time
                fps = frame_count / elapsed_time
                logging.info(f"Processed {frame_count} frames at {fps:.2f} FPS")

            # Stop after 30 seconds
            if time.time() - start_time > 30:
                break

    except Exception as e:
        logging.error(f"Error during testing: {str(e)}")
    finally:
        # Clean up
        cap.release()
        out.release()
        try:
            cv2.destroyAllWindows()
        except:
            pass
        logging.info("Recording completed. Output saved to hand_sign_detection_output.mp4")

if __name__ == "__main__":
    main() 