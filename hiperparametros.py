from sklearn.linear_model import LogisticRegression

# Crear el modelo con hiperparámetros ajustados
modelo = LogisticRegression(C=1.0, solver='liblinear', max_iter=100)

# Entrenar el modelo
modelo.fit(X_train, y_train)
