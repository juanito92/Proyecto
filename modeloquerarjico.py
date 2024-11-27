# Generar dendrograma
plt.figure(figsize=(10, 7))
linked = linkage(scaled_features, method='ward')
dendrogram(linked, truncate_mode='level', p=5)
plt.title('Dendrograma de Clustering Jerárquico')
plt.show()

# Cortar dendrograma para definir clústeres
hierarchical_labels = fcluster(linked, t=5, criterion='maxclust')

# Métricas de evaluación
silhouette_hierarchical = silhouette_score(scaled_features, hierarchical_labels)
calinski_hierarchical = calinski_harabasz_score(scaled_features, hierarchical_labels)
print(f"Silhouette Score (Jerárquico): {silhouette_hierarchical:.2f}")
print(f"Calinski-Harabasz Index (Jerárquico): {calinski_hierarchical:.2f}")

# Visualización de los clústeres jerárquicos
plt.figure(figsize=(8, 6))
sns.scatterplot(x=scaled_features[:, 0], y=scaled_features[:, 1], hue=hierarchical_labels, palette='viridis')
plt.title('Clústeres Jerárquicos')
plt.xlabel('Ingreso Anual (Estandarizado)')
plt.ylabel('Puntaje de Gasto (Estandarizado)')
plt.show()
