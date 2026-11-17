# TP — Bases de PyTorch

Travaux pratiques en français pour comprendre les tenseurs, la différentiation automatique et l'entraînement de réseaux de neurones avec PyTorch.

## Contenu

Le notebook `Activité pratique Bases de Pytorch.ipynb` conserve la progression pédagogique et les cellules du document fourni.

- Fondamentaux : tenseurs, CPU/CUDA/MPS, Autograd, graphe de calcul, `nn.Module`, activations, fonctions de coût, optimiseurs, mini-lots et modes `train()` / `eval()`.
- Démonstrations : régression linéaire, classification de points en 2D, métriques, frontière de décision et sauvegarde/rechargement des paramètres.
- Exercice 1 : régression multiple des salaires fictifs, séparation 80/20, standardisation sur l'entraînement, MAE en DH, interprétation des coefficients et nouvelle prédiction.
- Exercice 2 : classification Iris, normalisation sans fuite de données, réseau avec et sans couche cachée, courbes de perte, matrice de confusion et prédiction d'une nouvelle fleur.
- Exercice 3 : MNIST avec un réseau dense, entraînement pendant 5 époques, accuracy et métriques macro, matrice de confusion, erreurs illustrées et vérification du rechargement.
- Exercice 4 : MNIST avec un CNN et Dropout, entraînement pendant 3 époques, évaluation, visualisation de 10 erreurs et vérification du rechargement.

## Technologies

Python, PyTorch, torchvision, Matplotlib, scikit-learn, torcheval, torchmetrics, torchviz et Jupyter. Graphviz est nécessaire pour dessiner le graphe des opérations.

## Installation et exécution

Créez un environnement Python compatible avec PyTorch, puis installez les dépendances :

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Sous Windows, activez l'environnement avec `.venv\Scripts\activate`.

Installez également Graphviz et vérifiez que `dot` est dans le PATH :

```bash
# macOS (Homebrew)
brew install graphviz
# Ubuntu / Debian
sudo apt install graphviz
```

Lancez Jupyter depuis la racine du projet :

```bash
python -m jupyter lab
```

Ouvrez le notebook, sélectionnez le noyau de l'environnement puis utilisez **Restart Kernel and Run All Cells**. L'ordre des cellules est important. Une connexion Internet est nécessaire au premier chargement de MNIST ; les données sont téléchargées dans `data/`. L'entraînement utilise MPS, CUDA ou le CPU suivant la disponibilité et peut prendre plusieurs minutes.

Pour une exécution automatisée conservant le même fichier :

```bash
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 *.ipynb
```

Les poids `.pt` / `.pth` et le graphe PNG sont régénérés pendant l'exécution et ignorés par Git, comme les données, environnements, caches et sauvegardes locales. Les graines pseudo-aléatoires sont fixées ; les résultats peuvent varier selon le matériel et les versions. Les métriques de la démonstration 2D portent sur l'entraînement ; les exercices Iris et MNIST utilisent un jeu de test distinct. Les salaires sont entièrement fictifs.

## Vérification du TP

Exécution complète sous Python 3.12.16 : 50 cellules de code sur 50, sans erreur. Les 107 cellules et leur ordre sont conservés. Les chemins locaux présents dans les avertissements ont été anonymisés.

| Exercice | Résultat mesuré |
| --- | --- |
| Salaires | MAE : 381,64 DH ; prédiction du nouveau salarié : 14 201,94 DH |
| Iris | Accuracy : 100 % avec couche cachée, 90 % sans couche cachée, sur 30 fleurs de test |
| MNIST dense | Accuracy : 97,45 % sur 10 000 images de test |
| MNIST CNN | Accuracy : 98,80 % sur 10 000 images de test |

Le rechargement des paramètres préserve les prédictions des trois modèles sauvegardés. Ces valeurs correspondent à cette exécution et ne constituent pas une garantie pour d'autres environnements.
