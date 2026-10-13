import math

import numpy as np
import random
import time

def produit_scalaire(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError("Les listes doivent avoir la même taille.")
    return sum(x * y for x, y in zip(a, b))

def norme(a: list[float]) -> float:
    return math.sqrt(produit_scalaire(a, a))

def distance(a: list[float], b: list[float]) -> float:
    return norme([x - y for x, y in zip(a, b)])

def similarite_cosinus(a: list[float], b: list[float]) -> float:
    if norme(a) == 0 or norme(b) == 0:
        raise ValueError("Les vecteurs ne doivent pas être nuls.")
    return produit_scalaire(a, b) / (norme(a) * norme(b))

def normaliser(a: list[float]) -> list[float]:
    total = norme(a)
    if total == 0:
        raise ValueError("Le vecteur ne doit pas être nul.")
    norm = [x / total for x in a]
    return norm

def produit_scalaire_np(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b)

def norme_np(a: np.ndarray) -> float:
    return np.linalg.norm(a)

def distance_np(a: np.ndarray, b: np.ndarray) -> float:
    return np.linalg.norm(a - b)

def similarite_cosinus_np(a: np.ndarray, b: np.ndarray) -> float:
    n_a = norme_np(a)
    n_b = norme_np(b)
    if n_a == 0 or n_b == 0:
        raise ValueError("Les vecteurs ne doivent pas être nuls.")
    return np.dot(a, b) / (n_a * n_b)


CATALOGUE = {
    "Mustang": [45000, 1700, 450, 2],
    "Camaro": [42000, 1650, 455, 3],
    "F-150": [55000, 2100, 400, 1],
    "Corvette": [80000, 1530, 495, 1],
    "Wrangler": [38000, 1900, 270, 4],
}
def plus_proches(nom: str, catalogue: dict, k: int = 2, methode: str = "distance") -> list[str]:
    if nom not in catalogue:
        raise ValueError(f"{nom} n'est pas dans le catalogue.")
    if methode not in ["distance", "similarite"]:
        raise ValueError(f"Méthode {methode} non reconnue. Utilisez 'distance' ou 'similarite'.")
    reference = catalogue[nom]
    scores = []
    for autre_nom, autre_vecteur in catalogue.items():
        if autre_nom == nom :
            continue
        if methode == "distance":
            score = distance(reference, autre_vecteur)
        else:  # similarite
            score = similarite_cosinus(reference, autre_vecteur)
        scores.append((autre_nom, score))
    if methode == "distance":
        scores.sort(key=lambda x: x[1])
    else:
        scores.sort(key=lambda x: x[1], reverse=True)
    return [nom for nom, _ in scores[:k]]



if __name__ == "__main__":
    reference = "Mustang"

    print(f"Véhicules les plus proches de la {reference}")
    print(f"  par distance   : {plus_proches(reference, CATALOGUE, methode='distance')}")
    print(f"  par similarité : {plus_proches(reference, CATALOGUE, methode='similarite')}")

    print()
    print("Comparaison pour chaque véhicule du catalogue :")
    for nom in CATALOGUE:
        par_distance = plus_proches(nom, CATALOGUE, methode="distance")
        par_similarite = plus_proches(nom, CATALOGUE, methode="similarite")
        identique = "identique" if par_distance == par_similarite else "différent"
        print(f"  {nom:<10} distance {par_distance}  similarité {par_similarite}  ({identique})")

    a = [random.random() for _ in range(1_000_000)]
    b = [random.random() for _ in range(1_000_000)]

    debut = time.perf_counter()
    produit_scalaire(a, b)
    duree_pur = time.perf_counter() - debut
    print(f"Python pur : {duree_pur:.4f} s")

    a_np = np.array(a)
    b_np = np.array(b)

    debut_np = time.perf_counter()
    np.dot(a_np, b_np)
    duree_np = time.perf_counter() - debut_np
    print(f"NumPy : {duree_np:.4f} s")

    vitesse = duree_pur / duree_np
    print(f"NumPy est {vitesse:.1f} fois plus rapide !")