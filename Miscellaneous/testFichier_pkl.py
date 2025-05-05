import pickle

# Charger le fichier calibration.pkl
with open('Balises Fixes/Resultat Calibration/calibration.pkl', 'rb') as f:
    cameraMatrix, dist = pickle.load(f)

# Afficher les données
print("Camera Matrix:")
print(cameraMatrix)

print("\nDistortion Coefficients:")
print(dist)