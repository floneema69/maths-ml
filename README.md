# Maths-ML

Les maths du machine learning, codées à la main en Python puis vérifiées avec NumPy et scikit-learn.

Ce repo regroupe une semaine de travail sur les bases mathématiques du ML : vecteurs, matrices, descente de gradient, probabilités et statistiques. Chaque notion est d'abord écrite en Python pur, puis comparée à NumPy. La semaine se termine par une régression linéaire entraînée de zéro sur le dataset California Housing.

Ce projet fait partie de mon parcours pour devenir IA engineer.

## Contenu

| Jour | Thème | Fichier | Ce que j'y code |
| --- | --- | --- | --- |
| 1 | Vecteurs et similarité | `jour01_vecteurs.py` | produit scalaire, norme, distance, similarité cosinus, mini-moteur de recherche |
| 2 | Matrices | `jour02_matrices.py` | produit matriciel, transposée, couche dense, réseau 784 → 128 → 10 |
| 3 | Descente de gradient | `jour03_regression.py` | régression linéaire, boucle d'entraînement, vérification numérique du gradient |
| 4 | Probabilités | `jour04_probas.py` | simulations, théorème de Bayes, softmax stable, entropie croisée, température |
| 5 | Statistiques | `jour05_stats.py` | stats descriptives, valeurs aberrantes, corrélation, standardisation |
| Projet | California Housing | `projet_california.py` | analyse des données, préparation et régression sans modèle tout fait |

## Structure

```
maths-ml/
├── pyproject.toml
├── README.md
├── src/
│   └── maths_ml/
│       ├── __init__.py
│       ├── jour01_vecteurs.py
│       ├── jour02_matrices.py
│       ├── jour03_regression.py
│       ├── jour04_probas.py
│       ├── jour05_stats.py
│       └── projet_california.py
└── tests/
    ├── test_jour01.py
    ├── test_jour02.py
    ├── test_jour03.py
    ├── test_jour04.py
    └── test_jour05.py
```

## Installation

```bash
git clone https://github.com/floneema69/maths-ml.git
cd maths-ml
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -e ".[dev]"
```

## Utilisation

Lancer tous les tests :

```bash
pytest
```

Lancer le code d'une journée :

```bash
python -m maths_ml.jour01_vecteurs
```

## Organisation Git

Le repo suit Git Flow :

- `main` contient les versions stables
- `develop` reçoit le travail en cours
- une branche par journée, nommée `feature/AAAAMMJJ_FNA_description` (exemple : `feature/20261012_FNA_vecteurs`)

## Avancement

- [X] Jour 1 : vecteurs et similarité
- [ ] Jour 2 : matrices et couche dense
- [ ] Jour 3 : régression linéaire et descente de gradient
- [ ] Jour 4 : probabilités et softmax
- [ ] Jour 5 : statistiques et standardisation
- [ ] Projet California Housing

## Ce que j'ai appris

Chaque journée se termine par une question. Mes réponses :

**Jour 1.** Pourquoi des colonnes d'unités différentes (euros, kilos) posent problème dans un calcul de distance ?

Parce que les échelles de valeurs sont trop différentes : une grande unité (comme le prix en milliers d'euros) va écraser les plus petites (comme le poids en kilos) dans le calcul, faussant ainsi les résultats.

**Jour 2.** Pourquoi une activation non linéaire est indispensable dans un réseau de neurones ?

_À rédiger_

**Jour 3.** Pourquoi met-on l'erreur au carré plutôt que de prendre sa valeur absolue ?

_À rédiger_

**Jour 4.** Pourquoi un test fiable à 99 % peut donner un résultat positif peu fiable quand la maladie est rare ?

_À rédiger_

**Jour 5.** Pourquoi calcule-t-on la moyenne et l'écart-type sur le jeu d'entraînement seulement ?

_À rédiger_

## Ressources

- [3Blue1Brown](https://www.3blue1brown.com/topics/linear-algebra) : séries « Essence of Linear Algebra » et « Neural networks »
- [StatQuest](https://statquest.org/) : statistiques et ML expliqués pas à pas
- [Mathematics for Machine Learning](https://mml-book.github.io/) : livre de référence, PDF gratuit