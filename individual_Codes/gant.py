import pandas as pd
import matplotlib.pyplot as plt

# DataFrame con días enteros representando los días de inicio y fin (sin fechas específicas)
df = pd.DataFrame({
    'task': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L'],
    'start': [20, 24, 26, 31, 3, 7, 10, 14, 18, 23, 28, 30],  # Días de inicio como números enteros
    'end': [31, 28, 31, 8, 9, 18, 17, 22, 23, 1, 5, 5],  # Días de fin como números enteros
    'completion_frac': [1, 1, 1, 1, 1, 0.95, 0.7, 0.35, 0.1, 0, 0, 0]  # Fracción completada
})

# Calcular los días hasta el inicio y hasta el fin (como días enteros)
df['days_to_start'] = df['start'] - df['start'].min()
df['days_to_end'] = df['end'] - df['start'].min()

# Duración de la tarea (en días)
df['task_duration'] = df['days_to_end'] - df['days_to_start'] + 1  # Incluir también el día final

# Días de tarea completada
df['completion_days'] = df['completion_frac'] * df['task_duration']

# Mostrar el DataFrame
print(df)

# Graficar el gráfico de Gantt
plt.barh(y=df['task'], width=df['task_duration'], left=df['days_to_start'])
plt.xlabel('Días')
plt.ylabel('Tareas')
plt.title('Gráfico de Gantt')
plt.show()
