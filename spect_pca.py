from sklearn.decomposition import PCA
from sklearn.externals import joblib

x, y = joblib.load("features/3s_5_img_a_features.pkl")
pca = PCA(n_components=1000)
pca.fit(x)
joblib.dump(pca.components_, "features/3s_5_img_a_pca.pkl")

