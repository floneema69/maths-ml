import numpy as np
from maths_ml.jour01_vecteurs import norme_l2

def test_norme_l2():
    v = np.array([3.0, 4.0])
    assert norme_l2(v) == 5.0
