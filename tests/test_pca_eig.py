from unittest import TestCase
import numpy as np

from pcalib.pcalib import pca_eig


class TestPCAEIG(TestCase):
    """Testing PCA algorithm based on SVD."""

    def setUp(self):
        """Create sample dataset for 4 samples with 3 features per sample.
        Two features are high correlated, so, after PCA it should be an one-dimensional."""
        array = []
        for x in range(1, 5):
            array.append([x, x ** 2, np.random.randint(10)])

        self.x = np.array(array)

    def test_number_of_principal_components(self):
        """Check if we reduced number of features from 3 -> 1."""
        x_reduced, _ = pca_eig(self.x)
        self.assertEqual(x_reduced.shape[1], 1)

    def test_principal_components_range(self):
        """Check if # of pc <= # of features."""
        x_reduced, _ = pca_eig(self.x)
        self.assertLessEqual(x_reduced.shape[1], self.x.shape[1])
