import pandas as pd

# Cargar el dataset
data = pd.read_csv('Titanic-Dataset.csv')

# Ver las primeras filas
print(data.head())

# Resumen de información
print(data.info())

# Descripción estadística
print(data.describe())


# Ver el número de valores nulos por columna
print(data.isnull().sum())


import matplotlib.pyplot as plt
import seaborn as sns

# Distribución de edades
sns.histplot(data['Age'], kde=True)
plt.title('Distribución de Edad')
plt.show()

# Distribución de la tarifa
sns.histplot(data['Fare'], kde=True)
plt.title('Distribución de la Tarifa')
plt.show()

# Distribución de supervivencia
sns.countplot(x='Survived', data=data)
plt.title('Supervivencia')
plt.show()


# Relación entre género y supervivencia
sns.countplot(x='Survived', hue='Sex', data=data)
plt.title('Supervivencia por Género')
plt.show()

# Relación entre clase y supervivencia
sns.countplot(x='Survived', hue='Pclass', data=data)
plt.title('Supervivencia por Clase')
plt.show()

# Relación entre edad y supervivencia
sns.boxplot(x='Survived', y='Age', data=data)
plt.title('Supervivencia según Edad')
plt.show()


