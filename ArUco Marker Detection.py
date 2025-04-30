# import the necessary packages
import pickle
from imutils.video import VideoStream
import cv2
import numpy as np

# Load the camera calibration data
with open('Balises Fixes/cameraMatrix.pkl', 'rb') as f:
    cameraMatrix = pickle.load(f)
with open('Balises Fixes/dist.pkl', 'rb') as f:
    dist = pickle.load(f)

# Choose the correct camera index (0 for external, 1 for built-in)
#cap = cv2.VideoCapture(1) #--> For built-in camera
cap = cv2.VideoCapture(0) #--> For external camera

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error: Webcam not detected or cannot be opened.")
    exit()
else:
    print("Webcam activated successfully.")

# Define the real-world size of the ArUco marker (e.g., 5 cm side length)
marker_length = 0.037  # in meters (5 cm)

# Load the ArUco dictionary : 
# Test with a specific dictionary : aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
# Or can choose any of the following dictionaries : 
aruco_dict = {
	"DICT_4X4_50": cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
	"DICT_4X4_100": cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100),
	"DICT_4X4_250": cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250),
	"DICT_4X4_1000": cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000),
	"DICT_5X5_50": cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50),
	"DICT_5X5_100": cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100),
	"DICT_5X5_250": cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250),
	"DICT_5X5_1000": cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000),
	"DICT_6X6_50": cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50),
	"DICT_6X6_100": cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_100),
	"DICT_6X6_250": cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250),
	"DICT_6X6_1000": cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000),
	"DICT_7X7_50": cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_50),
	"DICT_7X7_100": cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_100),
	"DICT_7X7_250": cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_250),
	"DICT_7X7_1000": cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_1000),
	"DICT_ARUCO_ORIGINAL": cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL),
	"DICT_APRILTAG_16h5": cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16h5),
	"DICT_APRILTAG_25h9": cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_25h9),
	"DICT_APRILTAG_36h10": cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h10),
	"DICT_APRILTAG_36h11": cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
}

# Set detection parameters(in default) which can adjust these parameters for better detection
parameters = cv2.aruco.DetectorParameters()

# Detect Markers in Each Frame
while True:
    ret, frame = cap.read()     #--> Read the frame from the webcam
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)      #--> Convert to grayscale

    # initialize lists to store detected corners and IDs
    all_corners = []
    all_ids = []

    for dict_name, dictionary in aruco_dict.items():
    # Detect markers for the current dictionary
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)
    # If markers are detected, add them to the combined list
        if ids is not None:
            all_corners.extend(corners)
            all_ids.extend(ids)
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length, cameraMatrix, dist)

    #if len(all_ids) > 0:
    # Convert IDs to a single array (optional)
    #    all_ids = np.array(all_ids).flatten()
    # Draw all detected markers
    #cv2.aruco.drawDetectedMarkers(frame, all_corners, all_ids)

                # Loop through each detected marker within this dictionary
            for i in range(len(ids)):
                # Draw the marker and its axis
                cv2.aruco.drawDetectedMarkers(frame, corners)
                cv2.drawFrameAxes(frame, cameraMatrix, dist, rvecs[i], tvecs[i], 0.1)

                # Calculate the distance to the marker
                tvec = tvecs[i][0]
                distance = np.sqrt(tvec[0]**2 + tvec[1]**2 + tvec[2]**2)

                # Display the marker's ID and distance on separate lines
                text_id = f"Dict: {dict_name}, ID: {ids[i][0]}"
                text_distance = f"Distance: {distance:.2f} m"
                cv2.putText(frame, text_id, (10, 30 + i * 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, text_distance, (10, 60 + i * 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Print the information in the console
                print(f"Dictionary: {dict_name}")
                print(f"Marker ID: {ids[i][0]}")
                print(f"Translation Vector (tvec): {tvec}")
                print(f"Distance: {distance:.2f} m\n")

    # If no markers are detected
    if len(all_ids) == 0:
        cv2.putText(frame, "No markers detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

# Show the frame
    cv2.imshow('ArUco Detection', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything
cap.release()
cv2.destroyAllWindows()