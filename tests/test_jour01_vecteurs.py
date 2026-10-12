import numpy as np
import pytest

from maths_ml.jour01_vecteurs import (
    CATALOGUE,
    distance,
    distance_np,
    norme,
    norme_np,
    normaliser,
    plus_proches,
    produit_scalaire,
    produit_scalaire_np,
    similarite_cosinus,
    similarite_cosinus_np,
)


# Exercice 1 : produit scalaire

def test_produit_scalaire():
    assert produit_scalaire([1, 2, 3], [4, 5, 6]) == 32


def test_produit_scalaire_tailles_differentes():
    with pytest.raises(ValueError):
        produit_scalaire([1, 2], [3])


# Exercice 2 : norme

def test_norme_l2():
    assert norme([3.0, 4.0]) == 5.0
    assert norme([1.0, 2.0, 2.0]) == 3.0


# Exercice 3 : distance et similarité cosinus

def test_distance():
    assert distance([1.0, 1.0], [4.0, 5.0]) == pytest.approx(5.0)


def test_distance_vecteurs_identiques():
    assert distance([2.0, 3.0], [2.0, 3.0]) == 0.0


def test_distance_tailles_differentes():
    with pytest.raises(ValueError):
        distance([1.0, 2.0, 3.0], [1.0, 2.0])


def test_cosinus_meme_direction():
    assert similarite_cosinus([1.0, 2.0], [2.0, 4.0]) == pytest.approx(1.0)


def test_cosinus_perpendiculaires():
    assert similarite_cosinus([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosinus_opposes():
    assert similarite_cosinus([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_cosinus_vecteur_nul():
    with pytest.raises(ValueError):
        similarite_cosinus([0.0, 0.0], [1.0, 2.0])


# Exercice 4 : normaliser

def test_normaliser():
    assert normaliser([3.0, 4.0]) == pytest.approx([0.6, 0.8])


def test_normaliser_norme_egale_un():
    assert norme(normaliser([7.0, -2.0, 5.0])) == pytest.approx(1.0)


def test_normaliser_vecteur_nul():
    with pytest.raises(ValueError):
        normaliser([0.0, 0.0])


# Exercice 5 : versions NumPy

def test_produit_scalaire_np_identique():
    a = np.random.rand(10)
    b = np.random.rand(10)
    assert np.isclose(produit_scalaire(list(a), list(b)), produit_scalaire_np(a, b))


def test_norme_np_identique():
    a = np.random.rand(10)
    assert np.isclose(norme(list(a)), norme_np(a))


def test_distance_np_identique():
    a = np.random.rand(10)
    b = np.random.rand(10)
    assert np.isclose(distance(list(a), list(b)), distance_np(a, b))


def test_similarite_cosinus_np_identique():
    a = np.random.rand(10)
    b = np.random.rand(10)
    assert np.isclose(similarite_cosinus(list(a), list(b)), similarite_cosinus_np(a, b))


# Exercice 6 : moteur de recherche

def test_plus_proches_distance():
    assert plus_proches("Mustang", CATALOGUE) == ["Camaro", "Wrangler"]


def test_plus_proches_exclut_reference():
    assert "Mustang" not in plus_proches("Mustang", CATALOGUE, k=4)


def test_plus_proches_respecte_k():
    assert len(plus_proches("Mustang", CATALOGUE, k=3)) == 3


def test_plus_proches_nom_inconnu():
    with pytest.raises(ValueError):
        plus_proches("Twingo", CATALOGUE)


def test_plus_proches_methode_inconnue():
    with pytest.raises(ValueError):
        plus_proches("Mustang", CATALOGUE, methode="hasard")