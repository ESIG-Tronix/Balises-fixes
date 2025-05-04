# import the necessary packages
import pickle
from imutils.video import VideoStream
import cv2
import numpy as np

# Load the camera calibration data
def load_camera_calibration(calibration_path):
    with open(f'{calibration_path}/cameraMatrix.pkl', 'rb') as f:
        camera_matrix = pickle.load(f)
    with open(f'{calibration_path}/dist.pkl', 'rb') as f:
        dist_coeffs = pickle.load(f)
    return camera_matrix, dist_coeffs

# Choose the correct camera index (0 for external, 1 for built-in)

def initialize_camera(camera_index= 1): #--> For built-in camera
#def initialize_camera(camera_index= 0): #--> For external camera
    # Initialize the webcam
    cap = cv2.VideoCapture(camera_index)
    # Check if the webcam is opened successfully
    if not cap.isOpened():
        print("Error: Webcam not detected or cannot be opened.")
        exit()
    print("Webcam activated successfully.")
    return cap


def detect_markers(frame, aruco_dict, parameters, camera_matrix, dist_coeffs, marker_length):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    detected_markers = []
    ids_list = []  # List to store IDs for display purposes

    # Detect markers for the current dictionary
    for dict_name, dictionary in aruco_dict.items():
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)
        # If markers are detected, add them to the combined list
        if ids is not None:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)
            for i in range(len(ids)):
                detected_markers.append({
                    "dict_name": dict_name,
                    "id": ids[i][0],
                    "tvec": tvecs[i][0],
                    "rvec": rvecs[i],
                    "corners": corners[i]
                })
                ids_list.append(ids[i][0])  # Add the ID to the list                
    return detected_markers, ids_list

# Process each detected marker : display its information and draw it on the frame
def process_marker(marker, frame, camera_matrix, dist_coeffs, index):
    print("Detected Markers:", marker)
    dict_name = marker["dict_name"]
    marker_id = marker["id"]
    tvec = marker["tvec"]
    rvec = marker["rvec"]
    corners = marker["corners"]

    # Draw the marker and its axis
    cv2.aruco.drawDetectedMarkers(frame, [corners])
    cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.1)

    # Calculate the distance to the marker
    distance = np.sqrt(tvec[0]**2 + tvec[1]**2 + tvec[2]**2)

    # Display the marker's ID and distance on separate lines
    text_id = f"Dict: {dict_name}, ID: {marker_id}"
    text_distance = f"Distance: {distance:.2f} m"
    cv2.putText(frame, text_id, (10, 30 + index*60 ), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, text_distance, (10, 60 + index*60 ), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Print the information in the console
    print(f"Dictionary: {dict_name}")
    print(f"Marker ID: {marker_id}")
    print(f"Translation Vector (tvec): {tvec}")
    print(f"Distance: {distance:.2f} m\n")


# Main function to run the ArUco marker detection
def main():
    # Load camera calibration data
    calibration_path = 'Balises Fixes/Resultat Calibration'
    camera_matrix, dist_coeffs = load_camera_calibration(calibration_path)

    # Initialize the camera
    cap = initialize_camera(camera_index=1)

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
        ret, frame = cap.read()  # Read the frame from the webcam
        if not ret:
            break

        # Detect markers
        detected_markers = detect_markers(frame, aruco_dict, parameters, camera_matrix, dist_coeffs, marker_length)

        # Process each detected marker
        if detected_markers:
            for index, marker in enumerate(detected_markers):
                process_marker(marker, frame, camera_matrix, dist_coeffs, index)
        else:
            cv2.putText(frame, "No markers detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Show the frame
        cv2.imshow('ArUco Detection', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release everything
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()