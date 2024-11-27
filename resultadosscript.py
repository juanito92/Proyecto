# Guardar etiquetas generadas
df['KMeans_Cluster'] = kmeans_labels
df['Hierarchical_Cluster'] = hierarchical_labels
df.to_csv("outputs/clustered_data.csv", index=False)

# Guardar gráficos
plt.savefig("outputs/plots/kmeans_clusters.png")
plt.savefig("outputs/plots/hierarchical_clusters.png")
