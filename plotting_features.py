import classify_features
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt


def produce_plot():
    #code taken from https://stackoverflow.com/questions/54939424/plotting-vectorized-text-documents-in-matplotlib
    X, cv, tf_transformer, *_ = classify_features.vectorized_training_data()

    NUMBER_OF_CLUSTERS = 2
    km = KMeans(
        n_clusters=NUMBER_OF_CLUSTERS, 
        init='k-means++', 
        max_iter=500)

    km.fit(X)
    clusters = km.predict(X)

    pca = PCA(n_components=2)
    two_dim = pca.fit_transform(X)

    scatter_x = two_dim[:, 0]
    scatter_y = two_dim[:, 1]
    plt.style.use('ggplot')

    fig, ax = plt.subplots()
    fig.set_size_inches(15,8)

    cmap = {0: 'green', 1: 'blue'}
    for group in np.unique(clusters):
        print(group)
        ix = np.where(clusters == group)
        ax.scatter(scatter_x[ix], scatter_y[ix], c=cmap[group], label=group)

    ax.legend()
    plt.xlabel("PCA 0")
    plt.ylabel("PCA 1")
    plt.figure(figsize=(3,3))
    plt.show()

if __name__ == '__main__':
    produce_plot()