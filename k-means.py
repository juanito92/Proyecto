# Importar las librerías necesarias
import pandas as pd # Manejo y análisis de estructuras de datos
import numpy as np # Cálculo numérico y análisis de datos
import seaborn as sns # Creación gráficos estadísticos
import matplotlib.pyplot as plt # Creación de gráficos 2D
import plotly.graph_objs as go # Gráficos en 3D
import plotly.express as px # Gráficos interactivos en 3D
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import warnings

# Para evitar advertencias de memoria en Windows con KMeans
warnings.filterwarnings('ignore')

# Cargar los datos
Datos = pd.read_csv('Mall_Customers.csv')

# Mostrar las primeras filas para análisis exploratorio
print(Datos.head())

# Análisis descriptivo de los datos
print(Datos.describe())

# Identificación de datos faltantes
print(Datos.isnull().sum())

# Identificación de datos atípicos
plt.figure(figsize=(8, 6))
sns.boxplot(x=Datos['Annual Income (k$)'])
plt.title('Identificación de Datos Atípicos')
plt.show()

# Reemplazar ceros por la media en la columna "Annual Income (k$)"
Datos['Annual Income (k$)'] = Datos['Annual Income (k$)'].replace(0, Datos['Annual Income (k$)'].mean())

# Filtrar y eliminar datos atípicos
nivel_minimo = 0
nivel_maximo = 100
Datos = Datos[(Datos['Annual Income (k$)'] > nivel_minimo) & (Datos['Annual Income (k$)'] < nivel_maximo)]

# Revisión final de la información
print(Datos.info())

# Seleccionar las características relevantes para el análisis
X = Datos[["Age", "Annual Income (k$)", "Spending Score (1-100)"]]

# Normalizar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Visualización previa del clustering
plt.scatter(X_scaled[:, 1], X_scaled[:, 2])
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Distribución de los datos')
plt.show()

# Determinación de la cantidad óptima de clusters usando el método del codo
Nc = range(1, 11)
kmeans = [KMeans(n_clusters=i) for i in Nc]
score = [kmeans[i].fit(X_scaled).inertia_ for i in range(len(kmeans))]

plt.plot(Nc, score, marker='o')
plt.xlabel('Número de Clusters')
plt.ylabel('Inercia')
plt.title('Método del Codo')
plt.show()

# Entrenamiento con KMeans con 5 clusters
modelo_kmeans = KMeans(n_clusters=5, random_state=42)
modelo_kmeans.fit(X_scaled)

# Agregar la columna "Grupo" al dataset con el número del cluster
Datos['Grupo'] = modelo_kmeans.labels_

# Evaluar el desempeño del modelo utilizando métricas
sil_score = silhouette_score(X_scaled, Datos['Grupo'])
calinski_score = calinski_harabasz_score(X_scaled, Datos['Grupo'])
davies_score = davies_bouldin_score(X_scaled, Datos['Grupo'])

# Mostrar métricas
print(f"Coeficiente de Silhouette: {sil_score:.3f}")
print(f"Índice de Calinski-Harabasz: {calinski_score:.3f}")
print(f"Índice de Davies-Bouldin: {davies_score:.3f}")

# Visualización de los resultados del clustering en 2D
plt.scatter(X_scaled[:, 1], X_scaled[:, 2], c=Datos['Grupo'], cmap='viridis')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Clientes agrupados por KMeans')
plt.show()

# Visualización en 3D de los resultados
grafica_3d = px.scatter_3d(Datos, x='Annual Income (k$)', y='Spending Score (1-100)', z='Age',
                           color='Grupo', opacity=0.7, title='Modelo KMeans con 5 Clusters')
grafica_3d.show()
