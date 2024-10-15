import cv2
import numpy as np
import time
from picamera2 import Picamera2
from datetime import datetime

# Load the pre-trained deep learning face detector model from OpenCV
configFile = "MobileNetSSD_deploy.prototxt"  # Path to the deploy.prototxt file
modelFile = "mobilenet_iter_73000.caffemodel"  # Path to the caffemodel file

net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

# Initialize the list of class labels MobileNet SSD was trained to detect
CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

# Define a function to process the video stream and detect humans
def detect_humans(frame):
    detected = False
    dim = (400, 400)
    
    # Resize the image to 400x400 pixels
    resized_frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    # Prepare the image for the deep learning model
    blob = cv2.dnn.blobFromImage(resized_frame, 0.007843, (300, 300), 127.5)
    
    # Pass the blob through the network and obtain the detections
    net.setInput(blob)
    detections = net.forward()
    
    # Loop over the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections by ensuring the confidence is greater than a threshold
        if confidence > 0.2:
            idx = int(detections[0, 0, i, 1])

            # If the detected object is a person
            if CLASSES[idx] == "person":
                detected = True
                break  # Stop further detections once a person is found

    return detected

def capture_image(picam2, filename):
    picam2.capture_file(filename)
    print(f"Captured {filename}")
    return cv2.imread(filename)

def main():
    # Initialize two cameras
    picam1 = Picamera2(0)  # First camera
    picam2 = Picamera2(1)  # Second camera

    # Configure both cameras for still image capture
    camera1_config = picam1.create_still_configuration()
    camera2_config = picam2.create_still_configuration()

    # Apply the configurations
    picam1.configure(camera1_config)
    picam2.configure(camera2_config)

    # Start both cameras
    picam1.start()
    picam2.start()

    try:
        while True:
            # Capture images from both cameras
            frame1 = capture_image(picam1, "camera1_image.jpg")
            frame2 = capture_image(picam2, "camera2_image.jpg")

            # Detect person in both frames
            detected_in_cam1 = detect_humans(frame1)
            detected_in_cam2 = detect_humans(frame2)

            # Print results
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if detected_in_cam1:
                print(f"{timestamp}: Detected in Camera 1.")
            elif detected_in_cam2:
                print(f"{timestamp}: Detected in Camera 2.")
            else:
                print(f"{timestamp}: No detection.")

            # Show the images in separate windows for each camera


            resized_image1 = cv2.resize(frame1, (300,300), interpolation=cv2.INTER_AREA)
            resized_image2 = cv2.resize(frame2, (300,300), interpolation=cv2.INTER_AREA)

            cv2.imshow('Camera 1', resized_image1)
            cv2.imshow('Camera 2', resized_image2)


            # Wait for 1ms to allow the window to update, break if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Add delay to avoid excessive captures
            time.sleep(1)

    except KeyboardInterrupt:
        print("Camera stopped")
    finally:
        # Stop both cameras gracefully and close OpenCV windows
        picam1.stop()
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
