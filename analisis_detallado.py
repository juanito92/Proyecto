from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Calcular las métricas
# La precisión mide cuántas de las predicciones positivas fueron correctas
precision = precision_score(y_test, y_pred)

# El recall mide cuántos de los verdaderos positivos fueron identificados correctamente
recall = recall_score(y_test, y_pred)

# El F1-score es la media armónica entre la precisión y el recall, combinando ambas métricas
f1 = f1_score(y_test, y_pred)

# Imprimir las métricas
print(f'Precisión: {precision}')  # El valor de precisión
print(f'Recall: {recall}')        # El valor de recall
print(f'F1-score: {f1}')          # El valor de F1-score

# Generar la matriz de confusión para entender los errores del modelo
# La matriz de confusión muestra el número de falsos positivos, falsos negativos, verdaderos positivos y verdaderos negativos
cm = confusion_matrix(y_test, y_pred)

# Graficar la matriz de confusión usando seaborn para una mejor visualización
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicho No', 'Predicho Sí'], yticklabels=['Real No', 'Real Sí'])
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.title('Matriz de Confusión')
plt.show()

# Mostrar el reporte de clasificación completo con precisión, recall, F1-score para cada clase
print("\nReporte de clasificación:\n")
print(classification_report(y_test, y_pred))

# Interpretación de los resultados basada en las métricas obtenidas
print("\nInterpretación de resultados:")
if precision > 0.8 and recall > 0.8:
    print("El modelo tiene un buen rendimiento en términos de precisión y recall, indicando que está clasificando correctamente la mayoría de los casos.")
elif precision < 0.5:
    print("El modelo tiene una precisión baja, lo que sugiere que está cometiendo muchos errores en la clasificación positiva.")
else:
    print("El modelo podría necesitar mejoras en alguna de las métricas para lograr un mejor balance entre precisión y recall.")
