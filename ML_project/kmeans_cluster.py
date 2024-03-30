from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import numpy as np

def kmeans_cluster():
    # Load data
    train_data = np.load('data/train_data.npy')

    silhouette_scores = {}
    K = range(5, 17)

    for k in K:
        print(k)
        model = KMeans(n_clusters=k, random_state=42)
        cluster_labels = model.fit_predict(train_data)
        silhouette_avg = silhouette_score(train_data, cluster_labels)
        print(silhouette_avg)
        silhouette_scores[k] = silhouette_avg
    
    # Find the best k and highest score
    best_k = max(silhouette_scores, key=silhouette_scores.get)
    best_silhouette_score = silhouette_scores[best_k]

    return best_k, best_silhouette_score


