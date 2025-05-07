#  Balises Fixes – Système de détection et localisation de marqueurs ArUco sur l'aire de jeu (Coupe de France de Robotique)

## Description

Le projet **Balises Fixes** permet la détection et la localisation de **marqueurs ArUco** sur l'aire de jeu de la Coupe de France de Robotique. À l'aide d'une caméra, il identifie les marqueurs et estime leur position et orientation en utilisant un algorithme de pose. Ce système permet de suivre précisément les mouvements des robots principales sur le terrain.

## Dépendances

Le projet utilise les bibliothèques suivantes :

- **OpenCV 3.3+** 
- **Numpy**
- **Imutils** : Gestion du flux vidéo.
- **Scipy** : Manipulation des rotations 3D.
- **Pickle** : Chargement des données de calibration.

### Installation des dépendances

Pour installer les bibliothèques nécessaires à l'exécution du projet, il suffit d'exécuter la commande suivante dans le terminal :

```bash
pip install numpy opencv-contrib-python imutils scipy
