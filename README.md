# GlamFlow AI - Skin Type Analyzer

Cette application utilise l'intelligence artificielle pour analyser et déterminer votre type de peau (sèche, normale ou grasse) à partir d'une photo.

## Structure du projet

```
Backend/
├── run.py                    # API Flask pour les prédictions
├── saved_models/             # Dossier contenant les modèles entraînés
│   └── skin_best_model.pth   # Modèle pour la classification des types de peau
└── frontend/                 # Composants React pour l'interface utilisateur
    ├── SkinTypeAnalyzer.js   # Composant principal pour l'analyse de la peau
    └── SkinTypeAnalyzer.css  # Styles pour le composant
```

## Configuration de l'environnement

### Backend (Flask)

1. Installer les dépendances Python :

```bash
pip install flask flask-cors torch torchvision timm pillow numpy
```

2. Lancer le serveur Flask :

```bash
cd Backend
python run.py
```

Le serveur API sera accessible à l'adresse : http://localhost:5000

### Frontend (React)

1. Intégrer les fichiers dans votre application React existante :
   - Copier `SkinTypeAnalyzer.js` et `SkinTypeAnalyzer.css` dans votre dossier de composants
   - Importer et utiliser le composant dans votre application

2. Installer les dépendances nécessaires :

```bash
npm install axios
```

## Utilisation de l'API

L'API expose les endpoints suivants :

- `GET /health` : Vérifier que l'API est fonctionnelle
- `POST /predict` : Analyser une image pour déterminer le type de peau

Exemple de requête avec curl :

```bash
curl -X POST -F "file=@path/to/your/image.jpg" http://localhost:5000/predict
```

## Fonctionnalités du composant React

Le composant SkinTypeAnalyzer offre :

- Upload d'image depuis l'appareil
- Prise de photo via la caméra
- Affichage des résultats d'analyse avec les probabilités pour chaque type de peau

## Modèle d'IA utilisé

Le modèle utilise l'architecture REXNet-150 entraîné pour classifier les types de peau en trois catégories :
- Sèche (dry)
- Normale (normal)
- Grasse (oily)
