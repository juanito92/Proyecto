# Cargar el dataset
df = pd.read_csv("data/Mall_Customers.csv")

# Vista inicial de los datos
print(df.head())
print(df.info())

# Resumen estadístico
print(df.describe())

# Distribuciones iniciales
plt.figure(figsize=(10, 5))
sns.histplot(df['Age'], kde=True, bins=20, color='blue', label='Age')
plt.title('Distribución de Edades')
plt.legend()
plt.show()
