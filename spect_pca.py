import numpy as np

from sklearn.decomposition import PCA
from sklearn.externals import joblib

VARIANCE_P = 0.95

xo, y = joblib.load("processed_data/xy_3s_5_img.pkl")
x = xo.reshape(xo.shape[0], -1)
xbar = np.mean(x, axis=0)
x = x - xbar

# pca = PCA(n_components=VARIANCE_P)
# pca.fit(x)
# joblib.dump((pca.components_, xbar), "features/3s_5_img_pca.pkl")
# print(pca.components_.shape)

pca_load, xbar = joblib.load('features/3s_5_img_pca.pkl')

print x.shape, xbar.shape, pca_load.shape

# xhat = xbar + np.dot(x, pca.components_.T)
z = np.dot(x, pca_load.T)
print z.shape
xhat = np.dot(z, pca_load) + xbar
# xn = xhat[1, :]
# xn = xn.reshape((257, 257))
xhat = xhat[4, :].reshape((257, 257))
# x = xo[0, :]
joblib.dump(xhat, 'pca.pkl')
