import pandas as pd

# Cargar datos
df = pd.read_csv("data/Mall_Customers.csv")
df.head()

df.describe()

import matplotlib.pyplot as plt
import seaborn as sns

# Distribución de la edad
sns.histplot(df['Age'], kde=True)
plt.title('Distribución de Edad')
plt.show()

correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Matriz de Correlaciones")
plt.show()

sns.boxplot(x=df['Annual Income (k$)'])
plt.title("Valores Atípicos en Ingreso Anual")
plt.show()

