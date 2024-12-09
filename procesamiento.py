import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Cargar datos
df = pd.read_csv('tus_datos.csv')

# Rellenar valores faltantes con la media
imputer = SimpleImputer(strategy='mean')
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Convertir columnas categóricas a numéricas
df = pd.get_dummies(df, drop_first=True)

# Normalizar los datos
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# Guardar el dataframe procesado
df_scaled.to_csv('tus_datos_procesados.csv', index=False)
