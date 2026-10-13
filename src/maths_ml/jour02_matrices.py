import numpy as np

from maths_ml.jour01_vecteurs import produit_scalaire  # AJOUTÉ


def produit_matrice_vecteur(M: list[list[float]], v: list[float]) -> list[float]:
    return [produit_scalaire(ligne, v) for ligne in M]


def transposee(A: list[list[float]]) -> list[list[float]]:
    return [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]


def produit_matriciel(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    if len(A[0]) != len(B):
        raise ValueError(
            f"Formes incompatibles : A a {len(A[0])} colonnes, B a {len(B)} lignes."
        )
    return [
        [sum(A[i][k] * B[k][j] for k in range(len(B))) for j in range(len(B[0]))]
        for i in range(len(A))
    ]


def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0, z)


def couche_dense(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    return relu(X @ W + b)


class ReseauSimple:

    def __init__(self):
        self.W1 = np.random.randn(784, 128) * 0.01
        self.b1 = np.zeros(128)
        self.W2 = np.random.randn(128, 10) * 0.01
        self.b2 = np.zeros(10)

    def forward(self, X: np.ndarray) -> np.ndarray:
        a1 = couche_dense(X, self.W1, self.b1)
        z2 = a1 @ self.W2 + self.b2
        return z2


if __name__ == "__main__":

    M = np.zeros((64, 128))
    print((M + np.ones(128)).shape)
    print((M + np.ones(64).reshape(64, 1)).shape)

    X = np.random.rand(64, 784)
    sortie = ReseauSimple().forward(X)
    print(f"Forme de la sortie du réseau : {sortie.shape}")