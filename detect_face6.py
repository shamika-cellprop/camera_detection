import cv2
import numpy as np
import time
from picamera2 import Picamera2
from datetime import datetime

# Load the pre-trained deep learning face detector model from OpenCV
configFile = "deploy.prototxt"  # Path to the deploy.prototxt file
modelFile = "res10_300x300_ssd_iter_140000.caffemodel"  # Path to the caffemodel file

net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

def detect_face(frame):
	#print("entered")
    dim = (300, 300)
    resized_frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    # Prepare the image for the deep learning model
    (h, w) = resized_frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(resized_frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    # Pass the blob through the network and obtain the detections and predictions
    net.setInput(blob)
    detections = net.forward()

    # Loop over the detections
    detected = False
    for i in range(0, detections.shape[2]):
        # Extract the confidence (i.e., probability) associated with the prediction
        confidence = detections[0, 0, i, 2]
        #print("the confidence is",confidence)
        # Filter out weak detections by ensuring the confidence is greater than a minimum threshold
        if confidence > 0.5:
            detected = True
            # Compute the (x, y)-coordinates of the bounding box for the face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Draw the bounding box around the face
            cv2.rectangle(resized_frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # Print timestamped detection message
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp}: Detected." if detected else f"{timestamp}: Not detected.")

    # Return the processed frame and detection status
    return resized_frame, detected

def capture_image(picam2):
    filename = 'faces1.jpg'
    picam2.capture_file(filename)
    print(f"Captured {filename}")
    return cv2.imread(filename)

def main():
    # Initialize the camera
    picam2 = Picamera2()

    # Configure the camera for still image capture
    camera_config = picam2.create_still_configuration()
    picam2.configure(camera_config)

    # Start the camera
    picam2.start()

    try:
        while True:
            # Capture the image every 0.1 seconds
            frame = capture_image(picam2)
            time.sleep(0.1)
            processed_frame, detected = detect_face(frame)
            
            print(f"Detected: {detected}")
            
            # Show the image in the window
            cv2.imshow('Face Detection', processed_frame)

            # Wait for 1ms to allow the window to update, break if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(1)
    except KeyboardInterrupt:
        print("Camera stopped")
    finally:
        # Stop the camera gracefully and close OpenCV windows
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
