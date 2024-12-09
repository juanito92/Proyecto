from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# Realizar predicciones
y_pred = modelo.predict(X_test)

# Evaluar el desempeño
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Precisión:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("Matriz de confusión:\n", conf_matrix)
