import numpy as np


def lancer_de(n: int, graine: int = 42) -> np.ndarray:
    """Simule n lancers d'un dé à 6 faces."""
    np.random.seed(graine)
    return np.random.randint(1, 7, size=n)


def tirer_normale(
    n: int, moyenne: float, ecart_type: float, graine: int = 42
) -> np.ndarray:
    """Tire n valeurs selon une loi normale de moyenne et d'écart-type donnés."""
    np.random.seed(graine)
    return np.random.normal(moyenne, ecart_type, size=n)


def proba_malade_si_positif(
    p_malade: float, p_positif_si_malade: float, p_positif_si_sain: float
) -> float:
    """Calcule la probabilité d'être malade quand le test est positif (Bayes)."""
    p_sain = 1 - p_malade
    p_positif = p_positif_si_malade * p_malade + p_positif_si_sain * p_sain
    return (p_positif_si_malade * p_malade) / p_positif


def simuler_depistage(
    n: int,
    p_malade: float,
    p_positif_si_malade: float,
    p_positif_si_sain: float,
    graine: int = 42,
) -> float:
    """Simule un dépistage sur n personnes et renvoie la part de malades parmi les positifs."""
    np.random.seed(graine)
    malades = np.random.rand(n) < p_malade
    positifs = np.where(
        malades,
        np.random.rand(n) < p_positif_si_malade,
        np.random.rand(n) < p_positif_si_sain,
    )
    return float(malades[positifs].mean())


def softmax(z: np.ndarray) -> np.ndarray:
    """Transforme des scores bruts en probabilités qui s'additionnent à 1."""
    exp_z = np.exp(z - np.max(z, axis=-1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)


def entropie_croisee(probas: np.ndarray, classe_vraie: int) -> float:
    """Calcule la perte d'entropie croisée pour une prédiction."""
    return float(-np.log(probas[classe_vraie] + 1e-12))


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from maths_ml.jour02_matrices import ReseauSimple

    lancers = lancer_de(10_000)
    faces, comptes = np.unique(lancers, return_counts=True)
    print(f"Faces   : {faces}")
    print(f"Comptes : {comptes}")
    print(f"Moyenne : {lancers.mean():.3f}")

    print("\nMoyennes pour différents nombres de lancers :")
    for n in [10, 100, 10_000, 1_000_000]:
        print(f"{n} lancers : moyenne = {lancer_de(n).mean():.3f}")

    valeurs = tirer_normale(100_000, 50, 10)
    plt.hist(valeurs, bins=100)
    plt.xlabel("Valeur")
    plt.ylabel("Nombre de tirages")
    plt.title("Loi normale : moyenne 50, écart-type 10")
    plt.show()

    proportion_1 = np.mean((valeurs > 40) & (valeurs < 60))
    proportion_2 = np.mean((valeurs > 30) & (valeurs < 70))
    print(f"\nEntre 40 et 60 : {proportion_1:.3f}")
    print(f"Entre 30 et 70 : {proportion_2:.3f}")

    p_formule = proba_malade_si_positif(0.001, 0.99, 0.02)
    p_simulation = simuler_depistage(1_000_000, 0.001, 0.99, 0.02)
    print(f"\nProbabilité d'être malade si positif (formule)    : {p_formule:.3f}")
    print(f"Probabilité d'être malade si positif (simulation) : {p_simulation:.3f}")

    print(f"\nSoftmax de [1, 2, 3]          : {softmax(np.array([1.0, 2.0, 3.0]))}")
    print(f"Softmax de [1000, 1001, 1002] : {softmax(np.array([1000.0, 1001.0, 1002.0]))}")

    print(f"\nPerte, bonne réponse à 90 %   : {entropie_croisee(np.array([0.9, 0.05, 0.05]), 0):.3f}")
    print(f"Perte, bonne réponse à 51 %   : {entropie_croisee(np.array([0.51, 0.49]), 0):.3f}")
    print(f"Perte, mauvaise réponse à 90 % : {entropie_croisee(np.array([0.9, 0.05, 0.05]), 1):.3f}")

    logits = ReseauSimple().forward(np.random.rand(64, 784))
    probas = softmax(logits)
    print(f"\nForme des probabilités : {probas.shape}")
    print(f"Probabilités de la première image : {np.round(probas[0], 3)}")
    print(f"Somme des 5 premières lignes : {np.round(probas.sum(axis=1)[:5], 3)}")