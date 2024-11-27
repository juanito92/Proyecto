from sklearn.metrics import silhouette_score

def calculate_silhouette(X, labels):
    return silhouette_score(X, labels)
