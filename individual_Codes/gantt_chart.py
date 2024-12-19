# gantt_chart.py

import pandas as pd
import matplotlib.pyplot as plt

def create_gantt_chart(tabla_tareas):
    """
    Procesa una tabla de tareas y genera un gráfico de Gantt.

    :param tabla_tareas: DataFrame con columnas ['Tarea', 'Ruta(i->j)', 'Di']
                         - 'Tarea': nombre de la tarea
                         - 'Ruta(i->j)': cadena con formato 'start -> end'
                         - 'Di': fracción completada de la tarea (como número)
    """
    # Crear el DataFrame con las columnas necesarias
    df = pd.DataFrame({
        'task': tabla_tareas['Tarea'],  # Columna con nombres de tareas
        'start': pd.to_numeric([ruta.split(' -> ')[0] for ruta in tabla_tareas['Ruta(i->j)']]),  # Convertir inicio a numérico
        'end': pd.to_numeric([ruta.split(' -> ')[1] for ruta in tabla_tareas['Ruta(i->j)']]),    # Convertir fin a numérico
        'completion_frac': pd.to_numeric(tabla_tareas['Di'])  # Convertir fracción completada a numérico
    })

    # Calcular días hasta el inicio y fin, duración de tareas y días completados
    df['days_to_start'] = df['start'] - df['start'].min()
    df['days_to_end'] = df['end'] - df['start'].min()
    df['task_duration'] = df['days_to_end'] - df['days_to_start'] + 1  # Duración en días incluyendo el final
    df['completion_days'] = df['completion_frac'] * df['task_duration']


    # Generar gráfico de Gantt
    plt.barh(y=df['task'], width=df['task_duration'], left=df['days_to_start'])
    plt.xlabel('Días')
    plt.ylabel('Tareas')
    plt.title('Gráfico de Gantt')
    plt.show()
