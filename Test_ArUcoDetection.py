

# ------------------------------------------------
# import the necessary packages
# ------------------------------------------------
import cv2
import numpy as np
import pickle

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
# Define reference coordinates for the ArUco markers
# ------------------------------------------------
reference_coords = {
    0: np.array([0.6, 0, 0.6]), 
    1: np.array([0.6, 0, 1.4]),  
    2: np.array([2.4, 0, 0.6]),  
    3: np.array([2.4, 0, 1.4])   
}

error_margin = 0.05  # 5 cm error margin

# ------------------------------------------------
# Start the webcam and detect markers
# ------------------------------------------------

cap = cv2.VideoCapture(0) # Open the default camera (0)

if not cap.isOpened():
    print("Error: Webcam not detected or cannot be opened.")
    exit()

# Marker length in meters
marker_length = 0.05  

# Define the ArUco dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()

# ------------------------------------------------
# Run the test
# ------------------------------------------------

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect ArUco markers
    corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    
    if ids is not None:
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length, cameraMatrix, dist)

        validation = True  # Assume all markers are validated initially

        for i in range(len(ids)):
            marker_id = ids[i][0]
            tvec = tvecs[i][0]
            
            # Calculate the distance (for reference purposes)
            distance = np.sqrt(tvec[0]**2 + tvec[1]**2 + tvec[2]**2)

            # Get the global coordinates of the marker
            detected_coords = tvec + np.array([1.5, -1, 0])  # Camera position offset
            print(f"Marker {marker_id} detected at coordinates: {detected_coords}")

            # Check if the detected coordinates are within the error margin of the reference coordinates
            reference_coord = reference_coords.get(marker_id)
            if reference_coord is not None:
                difference = np.linalg.norm(detected_coords - reference_coord)  

                if difference > error_margin:
                    validation = False
                    print(f"Marker {marker_id} not validated. Difference: {difference:.3f} meters.")
                else:
                    print(f"Marker {marker_id} validated. Difference: {difference:.3f} meters.")
            else:
                print(f"Marker {marker_id} not found in reference coordinates.")
                validation = False

        if validation:
            print("All markers validated successfully.")
        else:
            print("Not validated. Some markers have errors beyond the acceptable margin.")
    
    # Show the frame with marker detection
    cv2.imshow('ArUco Detection Test', frame)

    # Press 'q' to quit the test
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close any open windows
cap.release()
cv2.destroyAllWindows()
