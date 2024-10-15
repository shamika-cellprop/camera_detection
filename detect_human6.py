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

# Define a function to process the video stream
def detect_humans(frame):
	
	
    detected = False
    dim = (400, 400)
    print("entered")
    
    # Resize the image to 400x400 pixels
    resized_frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    # Prepare the image for the deep learning model
    (h, w) = resized_frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    
    # Pass the blob through the network and obtain the detections
    net.setInput(blob)
    detections = net.forward()
    
    # Loop over the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        #print("confidence value",confidence)

        # Filter out weak detections by ensuring the confidence is greater than a threshold
        if confidence > 0.2:
            idx = int(detections[0, 0, i, 1])

            # If the detected object is a person
            if CLASSES[idx] == "person":
                detected = True
                #box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                #(startX, startY, endX, endY) = box.astype("int")

                # Draw the bounding box around the detected person
                #cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                #label = f"{CLASSES[idx]}: {confidence:.2f}"
                #cv2.putText(frame, label, (startX, startY - 10),
                            #cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the output frame
    cv2.imshow("HumanFrame", resized_frame)
    
    return resized_frame, detected

def capture_image(picam2):
    filename = 'human1.jpg'
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
            time.sleep(1)
            processed_frame, detected = detect_humans(frame)
            
            #print(f"Detected: {detected}")
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"{timestamp}: Detected." if detected else f"{timestamp}: Not detected.")
            
            # Show the image in the window
            #cv2.imshow('human Detection', processed_frame)

            # Wait for 1ms to allow the window to update, break if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            #time.sleep(1)
    except KeyboardInterrupt:
        print("Camera stopped")
    finally:
        # Stop the camera gracefully and close OpenCV windows
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
