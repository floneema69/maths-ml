import numpy as np
import pytest

from maths_ml.jour02_matrices import ReseauSimple
from maths_ml.jour04_probas import (
    entropie_croisee,
    lancer_de,
    proba_malade_si_positif,
    simuler_depistage,
    softmax,
    softmax_temperature,
    tirer_normale,
)


def test_lancer_de_taille():
    assert lancer_de(500).shape == (500,)


def test_lancer_de_valeurs_entre_1_et_6():
    lancers = lancer_de(10_000)
    assert lancers.min() == 1
    assert lancers.max() == 6


def test_lancer_de_moyenne_proche_de_3_5():
    assert abs(lancer_de(100_000).mean() - 3.5) < 0.05


def test_tirer_normale_moyenne_et_ecart_type():
    valeurs = tirer_normale(100_000, 50, 10)
    assert abs(valeurs.mean() - 50) < 0.1
    assert abs(valeurs.std() - 10) < 0.1


def test_regle_des_68_pourcents():
    valeurs = tirer_normale(100_000, 50, 10)
    proportion = np.mean((valeurs > 40) & (valeurs < 60))
    assert abs(proportion - 0.68) < 0.01


def test_proba_malade_si_positif():
    assert proba_malade_si_positif(0.001, 0.99, 0.02) == pytest.approx(0.0472, abs=0.001)


def test_test_parfait():
    assert proba_malade_si_positif(0.001, 1.0, 0.0) == pytest.approx(1.0)


def test_simulation_proche_de_la_formule():
    formule = proba_malade_si_positif(0.001, 0.99, 0.02)
    simulation = simuler_depistage(1_000_000, 0.001, 0.99, 0.02)
    assert simulation == pytest.approx(formule, abs=0.01)


def test_softmax_valeurs():
    assert softmax(np.array([1.0, 2.0, 3.0])) == pytest.approx([0.090, 0.245, 0.665], abs=0.001)


def test_softmax_somme_a_un():
    assert softmax(np.array([0.5, -2.0, 7.0, 1.0])).sum() == pytest.approx(1.0)


def test_softmax_stable_avec_grandes_valeurs():
    petites = softmax(np.array([1.0, 2.0, 3.0]))
    grandes = softmax(np.array([1000.0, 1001.0, 1002.0]))
    assert np.allclose(petites, grandes)


def test_softmax_batch():
    probas = softmax(np.random.randn(64, 10))
    assert probas.shape == (64, 10)
    assert np.allclose(probas.sum(axis=1), 1)


def test_entropie_bonne_reponse_confiant():
    assert entropie_croisee(np.array([0.9, 0.05, 0.05]), 0) == pytest.approx(0.105, abs=0.001)


def test_entropie_bonne_reponse_hesitant():
    assert entropie_croisee(np.array([0.51, 0.49]), 0) == pytest.approx(0.673, abs=0.001)


def test_entropie_mauvaise_reponse_confiant():
    assert entropie_croisee(np.array([0.9, 0.05, 0.05]), 1) == pytest.approx(2.996, abs=0.001)


def test_entropie_proba_nulle_pas_infinie():
    assert np.isfinite(entropie_croisee(np.array([1.0, 0.0]), 1))


def test_softmax_sur_reseau():
    logits = ReseauSimple().forward(np.random.rand(64, 784))
    probas = softmax(logits)
    assert probas.shape == (64, 10)
    assert np.allclose(probas.sum(axis=1), 1)


def test_temperature_1_ne_change_rien():
    z = np.array([1.0, 2.0, 3.0])
    assert np.allclose(softmax_temperature(z, 1.0), softmax(z))


def test_temperature_basse_plus_confiante():
    z = np.array([1.0, 2.0, 3.0])
    assert softmax_temperature(z, 0.5).max() > softmax(z).max()


def test_temperature_haute_plus_uniforme():
    z = np.array([1.0, 2.0, 3.0])
    assert softmax_temperature(z, 2.0).max() < softmax(z).max()