import numpy as np
from sklearn.decomposition import PCA
from t2 import hotelling_tsquared
import pytest

# tsquared results are compared with mathworks example
#     http://www.mathworks.com/help/stats/pca.html#bti6r0c-1
#     https://stackoverflow.com/questions/25412954/hotellings-t2-scores-in-python

def test_hottellings_t2_values():
    hald_text = """Y       X1      X2      X3      X4
        78.5    7       26      6       60
        74.3    1       29      15      52
        104.3   11      56      8       20
        87.6    11      31      8       47
        95.9    7       52      6       33
        109.2   11      55      9       22
        102.7   3       71      17      6
        72.5    1       31      22      44
        93.1    2       54      18      22
        115.9   21      47      4       26
        83.8    1       40      23      34
        113.3   11      66      9       12
        109.4   10      68      8       12
        """

    expected = [
        5.68034084, 3.07583704, 6.00023279, 2.61976323, 3.36813945,
        0.56679667, 3.48184073, 3.9793979 , 2.60858624, 7.48175633,
        4.18302223, 2.23271872, 2.72156782]

    hald = np.loadtxt(hald_text.splitlines(), skiprows=1)
    ingredients = hald[:, 1:]
    pca = PCA().fit_transform(ingredients)
    result = hotelling_tsquared(pca)   

    assert len(expected) == len(result)
    assert np.allclose(expected, result)
