# Selección de características relevantes
features = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Estandarización
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Visualizar datos escalados
scaled_df = pd.DataFrame(scaled_features, columns=features.columns)
print(scaled_df.describe())
