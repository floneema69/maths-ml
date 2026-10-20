import numpy as np
from maths_ml.jour01_vecteurs import CATALOGUE, plus_proches

def moyenne(x: list[float]) -> float:
    """Calcule la moyenne d'une liste de valeurs."""
    return sum(x) / len(x)


def mediane(x: list[float]) -> float:
    """Calcule la médiane d'une liste de valeurs."""
    x_trie = sorted(x)
    n = len(x_trie)
    if n % 2 == 0:
        return (x_trie[n // 2 - 1] + x_trie[n // 2]) / 2
    return x_trie[n // 2]


def variance(x: list[float]) -> float:
    """Calcule la variance d'une liste de valeurs."""
    m = moyenne(x)
    return sum((xi - m) ** 2 for xi in x) / len(x)


def ecart_type(x: list[float]) -> float:
    """Calcule l'écart-type d'une liste de valeurs."""
    return variance(x) ** 0.5


def detecter_aberrantes(x: list[float]) -> list[float]:
    """Renvoie les valeurs aberrantes selon la méthode de l'écart interquartile (IQR)."""
    q1 = np.percentile(x, 25)
    q3 = np.percentile(x, 75)
    iqr = q3 - q1
    borne_basse = q1 - 1.5 * iqr
    borne_haute = q3 + 1.5 * iqr
    return [xi for xi in x if xi < borne_basse or xi > borne_haute]


def detecter_aberrantes_sigma(x: list[float]) -> list[float]:
    """Renvoie les valeurs à plus de 3 écarts-types de la moyenne."""
    m = moyenne(x)
    seuil = 3 * ecart_type(x)
    return [xi for xi in x if abs(xi - m) > seuil]


def correlation(x: list[float], y: list[float]) -> float:
    """Calcule le coefficient de corrélation de Pearson entre deux listes."""
    if len(x) != len(y):
        raise ValueError("Les listes doivent avoir la même taille.")
    s_x = ecart_type(x)
    s_y = ecart_type(y)
    if s_x == 0 or s_y == 0:
        raise ValueError("La corrélation n'est pas définie pour une liste de valeurs identiques.")
    m_x = moyenne(x)
    m_y = moyenne(y)
    covariance = sum((xi - m_x) * (yi - m_y) for xi, yi in zip(x, y)) / len(x)
    return covariance / (s_x * s_y)


class Standardiseur:
    """Met chaque colonne à une moyenne de 0 et un écart-type de 1."""

    def fit(self, X: np.ndarray) -> "Standardiseur":
        """Apprend la moyenne et l'écart-type de chaque colonne."""
        self.moyennes = np.mean(X, axis=0)
        self.ecarts_types = np.std(X, axis=0)
        self.ecarts_types[self.ecarts_types == 0] = 1
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Standardise X avec les moyennes et écarts-types appris."""
        return (X - self.moyennes) / self.ecarts_types

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Apprend sur X puis le standardise."""
        return self.fit(X).transform(X)


if __name__ == "__main__":
    x = [2, 4, 4, 4, 5, 5, 7, 9]
    print(f"Moyenne    : {moyenne(x)}")
    print(f"Médiane    : {mediane(x)}")
    print(f"Variance   : {variance(x)}")
    print(f"Écart-type : {ecart_type(x)}")

    donnees = [10, 12, 12, 13, 12, 11, 14, 13, 15, 10, 10, 100, 100]
    print(f"\nAberrantes (IQR)     : {detecter_aberrantes(donnees)}")
    print(f"Aberrantes (3 sigma) : {detecter_aberrantes_sigma(donnees)}")

    print(f"\nCorrélation positive : {correlation([1, 2, 3, 4, 5], [2, 4, 6, 8, 10]):.3f}")
    print(f"Corrélation négative : {correlation([1, 2, 3, 4, 5], [10, 8, 6, 4, 2]):.3f}")
    valeurs_x = list(np.linspace(-5, 5, 101))
    valeurs_y = [v**2 for v in valeurs_x]
    print(f"Corrélation de y = x² : {correlation(valeurs_x, valeurs_y):.3f}")

    X = np.random.rand(100, 4) * 50
    X_std = Standardiseur().fit_transform(X)
    print(f"\nMoyennes après standardisation    : {np.round(X_std.mean(axis=0), 3)}")
    print(f"Écarts-types après standardisation : {np.round(X_std.std(axis=0), 3)}")

    noms = list(CATALOGUE.keys())
    X = np.array(list(CATALOGUE.values()), dtype=float)
    X_std = Standardiseur().fit_transform(X)
    catalogue_std = dict(zip(noms, X_std.tolist()))
    print("\nPlus proches voisins, avant et après standardisation :")
    for nom in noms:
        avant = plus_proches(nom, CATALOGUE, k=2, methode="distance")
        apres = plus_proches(nom, catalogue_std, k=2, methode="distance")
        changement = "identique" if avant == apres else "changé"
        print(f"  {nom:<10} avant {avant}  après {apres}  ({changement})")