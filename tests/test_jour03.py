import numpy as np
import pytest

from maths_ml.jour03_regression import (
    entrainer,
    generer_donnees,
    gradient_numerique_b,
    gradient_numerique_w,
    gradients,
    mse,
    predire,
)


def test_generer_donnees_formes():
    x, y = generer_donnees(n=50)
    assert x.shape == (50,)
    assert y.shape == (50,)


def test_generer_donnees_reproductible():
    x1, y1 = generer_donnees()
    x2, y2 = generer_donnees()
    assert np.array_equal(x1, x2)
    assert np.array_equal(y1, y2)


def test_predire():
    assert np.array_equal(predire(np.array([1, 2]), 3, 2), np.array([5, 8]))


def test_mse():
    assert mse(np.array([1, 2, 3]), np.array([1, 2, 5])) == pytest.approx(4 / 3)


def test_mse_prediction_parfaite():
    y = np.array([1.0, 2.0, 3.0])
    assert mse(y, y) == 0.0


def test_gradients():
    dw, db = gradients(np.array([1.0, 2.0]), np.array([5.0, 8.0]), 0.0, 0.0)
    assert dw == pytest.approx(-21.0)
    assert db == pytest.approx(-13.0)


def test_gradients_nuls_au_minimum():
    x = np.array([0.0, 1.0, 2.0])
    y = 10 * x + 1
    dw, db = gradients(x, y, 10.0, 1.0)
    assert dw == pytest.approx(0.0)
    assert db == pytest.approx(0.0)


def test_gradient_w_correct():
    x, y = generer_donnees()
    dw, _ = gradients(x, y, 1.0, 0.5)
    assert dw == pytest.approx(gradient_numerique_w(x, y, 1.0, 0.5))


def test_gradient_b_correct():
    x, y = generer_donnees()
    _, db = gradients(x, y, 1.0, 0.5)
    assert db == pytest.approx(gradient_numerique_b(x, y, 1.0, 0.5))


def test_entrainer_retrouve_la_droite():
    x, y = generer_donnees()
    w, b, _ = entrainer(x, y)
    assert abs(w - 10) < 0.1
    assert abs(b - 1) < 0.1


def test_entrainer_perte_diminue():
    x, y = generer_donnees()
    _, _, historique = entrainer(x, y)
    assert historique[-1] < historique[0]