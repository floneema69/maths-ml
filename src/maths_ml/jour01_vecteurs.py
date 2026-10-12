import numpy as np

def norme_l2(vecteur: np.ndarray) -> float:
    """Calcule la norme L2 (euclidienne) d'un vecteur."""
    return float(np.linalg.norm(vecteur))

if __name__ == "__main__":
    v = np.array([3.0, 4.0])
    print(f"Vecteur: {v}, Norme L2: {norme_l2(v)}")
