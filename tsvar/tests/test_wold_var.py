import numpy as np
import tsvar
import scipy.special


def test_digamma():
    for _ in range(20):
        arr = np.random.random(size=(10, 10)) * 10
        val1 = tsvar.models._wold_var.digamma(arr)
        val2 = scipy.special.digamma(arr)
        assert np.allclose(val1, val2)


if __name__ == "__main__":

    test_digamma()
