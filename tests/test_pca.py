from unittest import TestCase
import numpy as np

from pcalib import pca_svd, pca_eig


class TestPCA(TestCase):
    """Testing PCA algorithm based on SVD."""

    def setUp(self):
        """Create sample dataset for 4 samples with 3 features per sample.
        Two features are high correlated, so, after PCA it should be an one-dimensional."""
        array = []
        for x in range(1, 5):
            array.append([x, x ** 2, np.random.randint(10)])

        self.x = np.array(array)

    def test_pca_methods_are_equal(self):
        """Check if reduced matrices are equal to 4 digit after point for two PCA methods."""
        x_reduced_svd, _ = pca_svd(self.x)
        x_reduced_eig, _ = pca_eig(self.x)
        np.testing.assert_almost_equal(x_reduced_eig, x_reduced_svd, decimal=5)
