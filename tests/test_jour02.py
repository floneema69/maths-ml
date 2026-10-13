import numpy as np
import pytest

from maths_ml.jour02_matrices import (
    ReseauSimple,
    couche_dense,
    produit_matrice_vecteur,
    produit_matriciel,
    relu,
    transposee,
)

A = [[1, 2], [3, 4]]
B = [[5, 6], [7, 8]]


# Exercice 1 : matrice × vecteur

def test_produit_matrice_vecteur():
    M = [[1, 2], [3, 4], [5, 6]]
    assert produit_matrice_vecteur(M, [1, 1]) == [3, 7, 11]


def test_produit_matrice_vecteur_tailles_differentes():
    with pytest.raises(ValueError):
        produit_matrice_vecteur([[1, 2], [3, 4]], [1, 2, 3])


# Exercice 2 : transposée

def test_transposee():
    assert transposee([[1, 2, 3], [4, 5, 6]]) == [[1, 4], [2, 5], [3, 6]]


def test_transposee_deux_fois():
    assert transposee(transposee(A)) == A


# Exercice 3 : produit matriciel

def test_produit_matriciel_carre():
    assert produit_matriciel(A, B) == [[19, 22], [43, 50]]


def test_produit_matriciel_non_carre():
    C = [[1, 2, 3], [4, 5, 6]]
    D = [[7, 8], [9, 10], [11, 12]]
    assert produit_matriciel(C, D) == [[58, 64], [139, 154]]


def test_produit_matriciel_comme_numpy():
    C = np.random.rand(3, 4)
    D = np.random.rand(4, 2)
    resultat = produit_matriciel(C.tolist(), D.tolist())
    assert np.allclose(resultat, C @ D)


def test_produit_matriciel_formes_incompatibles():
    with pytest.raises(ValueError):
        produit_matriciel([[1, 2, 3]], [[1, 2, 3]])


# Exercice 4 : propriétés

def test_produit_matriciel_pas_commutatif():
    assert produit_matriciel(B, A) == [[23, 34], [31, 46]]
    assert produit_matriciel(A, B) != produit_matriciel(B, A)


def test_transposee_du_produit():
    gauche = transposee(produit_matriciel(A, B))
    droite = produit_matriciel(transposee(B), transposee(A))
    assert gauche == droite


# Exercice 6 : couche dense

def test_relu():
    assert np.array_equal(relu(np.array([-2.0, 0.0, 3.0])), np.array([0.0, 0.0, 3.0]))


def test_couche_dense_forme():
    X = np.random.rand(64, 784)
    W = np.random.randn(784, 128) * 0.01
    b = np.zeros(128)
    assert couche_dense(X, W, b).shape == (64, 128)


def test_couche_dense_pas_de_negatif():
    X = np.random.rand(64, 784)
    W = np.random.randn(784, 128) * 0.01
    b = np.zeros(128)
    assert (couche_dense(X, W, b) >= 0).all()


# Exercice 7 : réseau à deux couches

def test_reseau_simple_formes_des_poids():
    reseau = ReseauSimple()
    assert reseau.W1.shape == (784, 128)
    assert reseau.b1.shape == (128,)
    assert reseau.W2.shape == (128, 10)
    assert reseau.b2.shape == (10,)


def test_reseau_simple_forme_sortie():
    sortie = ReseauSimple().forward(np.random.rand(64, 784))
    assert sortie.shape == (64, 10)


def test_reseau_simple_logits_peuvent_etre_negatifs():
    sortie = ReseauSimple().forward(np.random.rand(64, 784))
    assert (sortie < 0).any()