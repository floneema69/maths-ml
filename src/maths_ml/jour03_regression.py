import numpy as np


def generer_donnees(n: int = 100, graine: int = 42) -> tuple[np.ndarray, np.ndarray]:
    np.random.seed(graine)
    x = np.random.uniform(0, 1, n)
    y = 10 * x + 1 + np.random.normal(0, 0.1, n)
    return x, y


def predire(x: np.ndarray, w: float, b: float) -> np.ndarray:
    return w * x + b


def mse(y_pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((y_pred - y) ** 2))


def gradients(x: np.ndarray, y: np.ndarray, w: float, b: float) -> tuple[float, float]:
    erreur = predire(x, w, b) - y
    dw = 2 * np.mean(erreur * x)
    db = 2 * np.mean(erreur)
    return float(dw), float(db)


def gradient_numerique_w(
        x: np.ndarray, y: np.ndarray, w: float, b: float, eps: float = 1e-5
) -> float:
    perte_plus = mse(predire(x, w + eps, b), y)
    perte_moins = mse(predire(x, w - eps, b), y)
    return (perte_plus - perte_moins) / (2 * eps)

def gradient_numerique_b(
        x: np.ndarray, y: np.ndarray, w: float, b: float, eps: float = 1e-5
) -> float:
    perte_plus = mse(predire(x, w, b + eps), y)
    perte_moins = mse(predire(x, w, b - eps), y)
    return (perte_plus - perte_moins) / (2 * eps)

def entrainer(
        x: np.ndarray, y: np.ndarray, lr: float = 0.1, iterations: int = 1000
) -> tuple[float, float, list[float]]:
    w, b = 0.0, 0.0
    historique = []
    for _ in range(iterations):
        historique.append(mse(predire(x, w, b), y))
        dw, db = gradients(x, y, w, b)
        w -= lr * dw
        b -= lr * db
    return w, b, historique

def generer_donnees_multi(n: int = 200, graine: int = 42) -> tuple[np.ndarray, np.ndarray]:
    np.random.seed(graine)
    x1 = np.random.uniform(0, 1, n)
    x2 = np.random.uniform(0, 1, n)
    x3 = np.random.uniform(0, 1, n)
    y = 2 * x1 - 1 * x2 + 0.5 * x3 + 4 + np.random.normal(0, 0.1, n)
    X = np.column_stack((x1, x2, x3))
    return X, y


def entrainer_multi(
    X: np.ndarray, y: np.ndarray, lr: float = 0.1, iterations: int = 5000
) -> tuple[np.ndarray, float, list[float]]:
    n_features = X.shape[1]
    w = np.zeros(n_features)
    b = 0.0
    historique = []
    for _ in range(iterations):
        y_pred = X @ w + b
        historique.append(mse(y_pred, y))
        erreur = y_pred - y
        dw = 2 * (X.T @ erreur) / len(y)
        db = 2 * np.mean(erreur)
        w -= lr * dw
        b -= lr * db
    return w, b, historique

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    x, y = generer_donnees()

    plt.scatter(x, y)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Données autour de la droite y = 10x + 1")
    plt.show()

    print(f"MSE avec w = 0, b = 0  : {mse(predire(x, 0, 0), y):.4f}")
    print(f"MSE avec w = 10, b = 1 : {mse(predire(x, 10, 1), y):.4f}")

    w, b, _ = entrainer(x, y)
    print(f"Paramètres trouvés : w = {w:.2f}, b = {b:.2f}")

    with np.errstate(over="ignore", invalid="ignore"):
        for lr in [0.001, 0.1, 1.5]:
            _, _, historique = entrainer(x, y, lr=lr)
            print(f"lr = {lr} : perte finale = {historique[-1]}")
            plt.plot(historique, label=f"lr = {lr}")
    plt.ylim(1e-3, 1e3)
    plt.yscale("log")
    plt.xlabel("Itération")
    plt.ylabel("MSE")
    plt.title("Effet du learning rate")
    plt.legend()
    plt.show()

    X, y = generer_donnees_multi()
    w, b, _ = entrainer_multi(X, y)
    print(f"Mes poids    : {np.round(w, 2)}, b = {b:.2f}")

    from sklearn.linear_model import LinearRegression

    modele = LinearRegression().fit(X, y)
    print(f"scikit-learn : {np.round(modele.coef_, 2)}, b = {modele.intercept_:.2f}")
