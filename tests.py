from covid19 import utils, models
import numpy as np
import pytest


@pytest.mark.parametrize("lb,ub,alpha", [
    (1, 2, 0.5),
    (3, 6, 0.9),
    (0.1, 0.3, 0.99)
])
def test_utils_make_from_interval(lb, ub, alpha):
    assert np.isclose(
        (utils.make_lognormal_from_interval(lb, ub, alpha).interval(alpha)), (lb, ub)
    ).all()

    assert np.isclose(
        (utils.make_normal_from_interval(lb, ub, alpha).interval(alpha)), (lb, ub)
    ).all()


def test_models_seirbayes_default():
    model = models.SEIRBayes()
    S, E, I, R, t_space = model.sample(100)
    assert (S[0,] == 99).all()
    assert (E[0,] ==  1).all()
    assert (I[0,] ==  0).all()
    assert (R[0,] ==  0).all()
