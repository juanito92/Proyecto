# Verificar valores nulos
print("Valores nulos por columna:\n", df.isnull().sum())

# Relación entre ingresos y puntaje de gasto
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Gender')
plt.title('Relación entre Ingresos Anuales y Puntaje de Gasto')
plt.show()

# Matriz de correlación
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matriz de Correlación')
plt.show()
