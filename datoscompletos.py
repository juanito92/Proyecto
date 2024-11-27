# Librerías necesarias
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import silhouette_samples
import warnings

warnings.filterwarnings('ignore')

# 1. Análisis exploratorio
# Cargar el conjunto de datos
data = pd.read_csv('Mall_Customers.csv')

# Información básica
print(data.info())
print(data.describe())

# Visualización inicial: Distribuciones
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.histplot(data['Age'], kde=True, bins=15, color='skyblue')
plt.title('Distribución de Edad')
plt.xlabel('Edad')
plt.ylabel('Frecuencia')

plt.subplot(1, 2, 2)
sns.histplot(data['Annual Income (k$)'], kde=True, bins=15, color='salmon')
plt.title('Distribución de Ingreso Anual')
plt.xlabel('Ingreso Anual (k$)')
plt.ylabel('Frecuencia')

plt.tight_layout()
plt.show()

# Identificación de valores atípicos
plt.figure(figsize=(8, 4))
sns.boxplot(data['Annual Income (k$)'], color='gold')
plt.title('Boxplot: Ingreso Anual')
plt.show()

# Identificación de correlaciones
sns.pairplot(data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']], diag_kind='kde', palette='husl')
plt.show()

# 2. Preprocesamiento
# Filtrar columnas relevantes y escalar los datos
filtered_data = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
scaler = StandardScaler()
scaled_data = scaler.fit_transform(filtered_data)

# Verificar datos escalados
scaled_df = pd.DataFrame(scaled_data, columns=['Age', 'Annual Income', 'Spending Score'])
print(scaled_df.describe())

# 3. Construcción del modelo de clustering jerárquico
# Generar dendrograma
linked = linkage(scaled_data, method='ward')
plt.figure(figsize=(10, 7))
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=False)
plt.title('Dendrograma para Clustering Jerárquico')
plt.xlabel('Muestras')
plt.ylabel('Distancia')
plt.show()

# Aplicar clustering con número óptimo de clusters (determinado por el dendrograma)
num_clusters = 4  # Cambiar si se identifica otro número óptimo
model = AgglomerativeClustering(n_clusters=num_clusters, linkage='ward')
labels = model.fit_predict(scaled_data)
filtered_data['Cluster'] = labels

# 4. Evaluación del modelo
sil_score = silhouette_score(scaled_data, labels)
ch_score = calinski_harabasz_score(scaled_data, labels)
db_score = davies_bouldin_score(scaled_data, labels)

print(f"Coeficiente de Silhouette: {sil_score:.2f}")
print(f"Índice de Calinski-Harabasz: {ch_score:.2f}")
print(f"Índice de Davies-Bouldin: {db_score:.2f}")

# 5. Visualización de resultados
# Gráfico de clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=filtered_data['Annual Income (k$)'],
    y=filtered_data['Spending Score (1-100)'],
    hue=filtered_data['Cluster'],
    palette='Set1',
    s=100
)
plt.title('Clusters Identificados con Clustering Jerárquico')
plt.xlabel('Ingreso Anual (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend(title='Cluster')
plt.show()

# Gráfico de silueta
sample_silhouette_values = silhouette_samples(scaled_data, labels)
plt.figure(figsize=(8, 6))
plt.bar(range(len(sample_silhouette_values)), sample_silhouette_values, color='royalblue')
plt.axhline(sil_score, color='red', linestyle='--', label=f'Silhouette Avg = {sil_score:.2f}')
plt.title("Gráfico de Silueta")
plt.xlabel("Muestra")
plt.ylabel("Valor de Silueta")
plt.legend()
plt.show()
