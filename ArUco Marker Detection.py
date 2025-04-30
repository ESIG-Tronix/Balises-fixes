# import the necessary packages
import cv2
import numpy as np

# Choose the correct camera index (0 for external, 1 for built-in)
cap = cv2.VideoCapture(1)  # For built-in camera
# cap = cv2.VideoCapture(0)  # For external camera

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error: Webcam not detected or cannot be opened.")
    exit()
else:
    print("Webcam activated successfully.")

# Load the ArUco dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

# Set detection parameters (default)
parameters = cv2.aruco.DetectorParameters()

# Detect Markers in Each Frame
while True:
    ret, frame = cap.read()  # Read the frame from the webcam
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

    # Detect markers
    corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    # If markers are detected
    if ids is not None:
        # Draw detected markers
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # Display the marker IDs on the frame
        for i in range(len(ids)):
            text_id = f"ID: {ids[i][0]}"
            top_left = tuple(corners[i][0][0].astype(int))
            cv2.putText(frame, text_id, (top_left[0], top_left[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    else:
        # If no markers are detected
        cv2.putText(frame, "No markers detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Show the frame
    cv2.imshow('ArUco Detection', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything
cap.release()
cv2.destroyAllWindows()