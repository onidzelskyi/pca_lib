"""Sample of using PCA library"""

import pandas as pd

from pcalib import pca_svd, pca_eig


def main():
    df = pd.read_csv('price_csv/yahooBars_GLD_2.csv')
    x = df.ix[:, 1:].values

    x_reduced, _ = pca_svd(x)
    print('# of initial components: {}\n# of principal components: {}'.format(x.shape[1], x_reduced.shape[1]))

if __name__ == '__main__':
    main()
