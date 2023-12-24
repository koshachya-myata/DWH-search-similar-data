from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


def get_kmenas_metrics_over_k(vectors, min_k=2, max_k=30, step=1):
    distortions_sil = []
    distortions_elb = []
    K = range(min_k, max_k, step)
    for k in K:
        model = KMeans(n_clusters=k, random_state=42)
        model.fit(vectors)

        sil = silhouette_score(vectors, model.labels_, metric='euclidean')
        distortions_sil.append(sil)

        elb = model.inertia_
        distortions_elb.append(elb)
        print(f'{k}: {sil};;; {elb}')
    return distortions_sil, distortions_elb


def plot_metrics_grid(dstr, K, title):
    plt.figure(figsize=(16, 8))
    plt.plot(K, dstr, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title(title)
    plt.show()
