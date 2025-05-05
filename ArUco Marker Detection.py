
# ------------------------------------------------
# import the necessary packages
# ------------------------------------------------

import pickle
from imutils.video import VideoStream
import cv2
import numpy as np

# ------------------------------------------------
# Load the camera calibration data
# ------------------------------------------------

try:
    with open('Balises Fixes/Resultat Calibration/cameraMatrix.pkl', 'rb') as f:
        cameraMatrix = pickle.load(f)
except FileNotFoundError:
    print("Error: cameraMatrix.pkl not found. Check the file path.")
    exit()
try:
    with open('Balises Fixes/Resultat Calibration/dist.pkl', 'rb') as f:
        dist = pickle.load(f)
except FileNotFoundError:
    print("Error: dist.pkl not found. Check the file path.")
    exit()

# ------------------------------------------------
# Choose the correct camera index (depending on your system)
# ------------------------------------------------

cap = cv2.VideoCapture(0) #--> 0 for external camera, 1 for built-in camera

# -------------------------------------------------
# Check if the webcam is opened successfully
# -------------------------------------------------

if not cap.isOpened():
    print("Error: Webcam not detected or cannot be opened.")
    exit()
else:
    print("Webcam activated successfully.")

# -------------------------------------------------
# Define the real-world size of the ArUco marker (5cm in our case)
# -------------------------------------------------

marker_length = 0.037  #--> in meters

# -------------------------------------------------
# Load the ArUco dictionary for marker detection:
# Choose from the predefined dictionaries available: 
# --------------------------------------------------

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

# -------------------------------------------------
# Set detection parameters(default parameters can be adjusted for better detection)
# -------------------------------------------------
parameters = cv2.aruco.DetectorParameters()

# --------------------------------------------------
# Detect Markers in Each Frame
# --------------------------------------------------

while True:
    ret, frame = cap.read()     #--> Read the frame from the webcam
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)      #--> Convert to grayscale

    # Initialize lists to store detected corners and IDs
    all_corners = []
    all_ids = []

    for dict_name, dictionary in aruco_dict.items():

        # Detect markers for the current dictionary
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)

        # ---------------------------------------------------
        # If markers are detected, add them to the combined list
        # ---------------------------------------------------
        if ids is not None:
            all_corners.extend(corners)
            all_ids.extend(ids)
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length, cameraMatrix, dist)

            # ----------------------------------------------------
            # Loop through each detected marker within this dictionary
            # ----------------------------------------------------
            for i in range(len(ids)):

                # -------------------------------------------------
                # Draw the marker and its axis

                # The following block of code manually draws the ArUco marker's bounding box and ID on the frame.
                # However, we can use this much simpler command to do the same thing "cv2.aruco.drawDetectedMarkers(frame, corners)
                # -------------------------------------------------

                # Loop over the detected ArUco corners
                for (markerCorner, markerID) in zip(corners, ids):
                    # Extract the marker corners (which are always returned in
                    # top-left, top-right, bottom-right, and bottom-left order)
                    markerCorner = markerCorner[0]  # Get the 4 points for the current marker

                    # Now unpack the 4 points into (topLeft, topRight, bottomRight, bottomLeft)
                    (topLeft, topRight, bottomRight, bottomLeft) = markerCorner

                    # Convert each of the (x, y)-coordinate pairs to integers
                    topRight = (int(topRight[0]), int(topRight[1]))
                    bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                    bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                    topLeft = (int(topLeft[0]), int(topLeft[1]))

                    # Draw the bounding box of the ArUco detection
                    cv2.line(frame, topLeft, topRight, (0, 255, 0), 2)
                    cv2.line(frame, topRight, bottomRight, (0, 255, 0), 2)
                    cv2.line(frame, bottomRight, bottomLeft, (0, 255, 0), 2)
                    cv2.line(frame, bottomLeft, topLeft, (0, 255, 0), 2)

                    # Draw the ArUco marker ID on the image
                    cv2.putText(frame, str(markerID),
                                (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0), 2)
                    
                # Draw the 3D axis for each marker
                cv2.drawFrameAxes(frame, cameraMatrix, dist, rvecs[i], tvecs[i], 0.1)

                # ---------------------------------------------------
                # Calculate the distance to the marker
                # ---------------------------------------------------

                tvec = tvecs[i][0] 
                distance = np.sqrt(tvec[0]**2 + tvec[1]**2 + tvec[2]**2)

                # ---------------------------------------------------
                # Display the marker's ID and distance on separate lines
                # ---------------------------------------------------

                text_id = f"Dict: {dict_name}, ID: {ids[i][0]}"
                text_distance = f"Distance: {distance:.2f} m"
                cv2.putText(frame, text_id, (10, 30 + i * 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, text_distance, (10, 60 + i * 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # ---------------------------------------------------
                # Calculate the global coordinates
                # ---------------------------------------------------
                camera_position = np.array([1.5, -1, 0])    # --> Camera position in world frame (x=1.5, y=0, z=1) :
                    #   --> X = 1.5: Horizontal displacement.
                    #   --> Y = -1: 1 meter above the table (negative Y).
                    #   --> Z = 0: No vertical displacement.

                marker_global_coords = tvec + camera_position
                marker_global_x, marker_global_y, marker_global_z = marker_global_coords

                # ---------------------------------------------------
                # Print the information in the terminal
                # ---------------------------------------------------
                print(f"Dictionary: {dict_name}, Marker ID: {ids[i][0]}")
                print(f"Translation Vector (tvec): {tvec}")
                print(f"Distance: {distance:.2f}")
                print(f"World coordinates: [{marker_global_x:.4f}, {marker_global_y:.4f}, {marker_global_z:.4f}] \n")

    # ---------------------------------------------------
    # If no markers are detected
    # ---------------------------------------------------
    if len(all_ids) == 0:
        cv2.putText(frame, "No markers detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # -------------------------------------------------
    # Show the frame
    # -------------------------------------------------
    cv2.imshow('ArUco Detection', frame)

    # --------------------------------------------------
    # Press 'q' to quit
    # --------------------------------------------------
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -------------------------------------------------
# Release everything and close windows
# -------------------------------------------------
cap.release()
cv2.destroyAllWindows()