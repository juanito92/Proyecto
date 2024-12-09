import matplotlib.pyplot as plt

# Crear las métricas
metrics = [precision, recall, f1]
metric_names = ['Precisión', 'Recall', 'F1-score']

# Graficar
plt.bar(metric_names, metrics, color=['blue', 'orange', 'green'])
plt.title('Métricas del Modelo')
plt.xlabel('Métrica')
plt.ylabel('Valor')
plt.show()
