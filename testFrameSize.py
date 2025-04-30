import cv2
import glob

# Utiliser glob pour trouver toutes les images dans le répertoire
image_files = glob.glob('Balises Fixes/images/*.png')

if not image_files:
    print("Aucune image trouvée dans le répertoire spécifié.")
else:
    print(f"Nombre d'images trouvées : {len(image_files)}")
    for i, image_file in enumerate(image_files):
        # Charger chaque image
        img = cv2.imread(image_file)

        # Vérifier si l'image est chargée correctement
        if img is None:
            print(f"Impossible de charger l'image : {image_file}")
            continue

        # Obtenir la résolution
        h, w = img.shape[:2]
        print(f"Image {i + 1}: {image_file} - Résolution : Largeur = {w}, Hauteur = {h}")