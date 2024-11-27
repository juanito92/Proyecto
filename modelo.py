# Selección del número óptimo de clústeres usando el método del codo
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_features)
    inertia.append(kmeans.inertia_)

# Gráfica del método del codo
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Método del Codo')
plt.xlabel('Número de Clústeres')
plt.ylabel('Inercia')
plt.show()

# Entrenar el modelo con el número óptimo de clústeres
optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans_labels = kmeans.fit_predict(scaled_features)

# Métricas de evaluación
silhouette_kmeans = silhouette_score(scaled_features, kmeans_labels)
calinski_kmeans = calinski_harabasz_score(scaled_features, kmeans_labels)
print(f"Silhouette Score (K-Means): {silhouette_kmeans:.2f}")
print(f"Calinski-Harabasz Index (K-Means): {calinski_kmeans:.2f}")

# Visualización de los clústeres
plt.figure(figsize=(8, 6))
sns.scatterplot(x=scaled_features[:, 0], y=scaled_features[:, 1], hue=kmeans_labels, palette='viridis')
plt.title('Clústeres K-Means')
plt.xlabel('Ingreso Anual (Estandarizado)')
plt.ylabel('Puntaje de Gasto (Estandarizado)')
plt.show()
