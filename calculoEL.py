import pandas as pd

def calcular_early_times(nodos, relaciones, duraciones):
    """
    Calcula Early Times (Ei) para una red de nodos.
    
    :param nodos: Lista de nodos (ordenados según dependencias para asegurar cálculos progresivos).
    :param relaciones: Diccionario donde las claves son nodos y los valores son listas de tuplas (actividad, destino).
    :param duraciones: Diccionario con las duraciones de las actividades.
    :return: Diccionario con los tiempos Ei y un DataFrame con los resultados.
    """
    # Inicializar Early Times (Ei) en 0 para todos los nodos
    Ei = {nodo: 0 for nodo in nodos}
    
    # Calcular Early Times (Ei) nodo por nodo
    for nodo in nodos:
        if nodo in relaciones:  # Si el nodo tiene conexiones salientes
            for actividad, destino in relaciones[nodo]:
                # Actualizar Ei del destino como el máximo entre los valores actuales y el nuevo cálculo
                Ei[destino] = max(Ei[destino], Ei[nodo] + duraciones[actividad])
    
    # Crear un DataFrame con los resultados
    tabla_early = pd.DataFrame({
        'Ti': nodos,
        'Ei': [Ei[nodo] for nodo in nodos],
    })
    
    return Ei, tabla_early

def calcular_late_times(nodos, relaciones, duraciones, Ei):
    """
    Calcula Late Times (Li) para una red de nodos, usando la lógica inversa de los Early Times.
    
    :param nodos: Lista de nodos (ordenados según dependencias para asegurar cálculos progresivos).
    :param relaciones: Diccionario donde las claves son nodos y los valores son listas de tuplas (actividad, destino).
    :param duraciones: Diccionario con las duraciones de las actividades.
    :param Ei: Diccionario con los Early Times calculados previamente.
    :return: Diccionario con los tiempos Li y un DataFrame con los resultados.
    """
    # Inicializar Late Times (Li) con valores grandes (un valor suficientemente grande para empezar)
    Li = {nodo: float('inf') for nodo in nodos}
    
    # El último nodo tiene Li igual a Ei
    Li[nodos[-1]] = Ei[nodos[-1]]
    
    # Calcular Late Times (Li) de forma inversa
    for nodo in reversed(nodos[:-1]):
        if nodo in relaciones:  # Si el nodo tiene conexiones salientes
            for actividad, destino in relaciones[nodo]:
                # Actualizar Li del nodo como el mínimo entre los valores actuales y el cálculo usando la actividad
                Li[nodo] = min(Li[nodo], Li[destino] - duraciones[actividad])
    
    # Crear un DataFrame con los resultados
    tabla_late = pd.DataFrame({
        'Ti': nodos,
        'Li': [Li[nodo] for nodo in nodos],
    })
    
    return Li, tabla_late

def generar_tabla_tareas(nodos, relaciones, duraciones, Ei, Li):
    """
    Genera la tabla de tareas con la información de la ruta, duración, Ei, Lj, Hij y estado crítico.
    
    :param nodos: Lista de nodos.
    :param relaciones: Diccionario de relaciones entre nodos.
    :param duraciones: Diccionario con las duraciones de las actividades.
    :param Ei: Diccionario con los tiempos de inicio temprano (Ei) de cada nodo.
    :param Li: Diccionario con los tiempos de inicio tardío (Li) de cada nodo.
    :return: DataFrame con la tabla de tareas.
    """
    # Lista de resultados para la tabla
    tareas_info = []
    
    # Recorremos todas las actividades para llenar la tabla
    for nodo in nodos:
        if nodo in relaciones:  # Si el nodo tiene conexiones salientes
            for actividad, destino in relaciones[nodo]:
                # Obtener la duración de la tarea
                Di = duraciones[actividad]
                # Obtener Ei para el nodo origen y Lj para el nodo destino
                Ei_val = Ei[nodo]
                Lj_val = Li[destino]
                # Calcular Hij (holgura)
                Hij = Lj_val - Di - Ei_val
                # Verificar si la tarea es crítica
                critico = "Sí" if Hij == 0 else "No"
                # Añadir la fila correspondiente a la tabla
                tareas_info.append([actividad, f"{nodo} -> {destino}", Di, Ei_val, Lj_val, Hij, critico])
    
    # Crear un DataFrame con la información recopilada
    tabla_tareas = pd.DataFrame(tareas_info, columns=["Tarea", "Ruta(i->j)", "Di", "Ei", "Lj", "Hij", "Critico"])
    
    return tabla_tareas


# Nodos en orden progresivo de cálculo
nodos = [1, 2, 3, 4]

# Relaciones entre nodos (origen -> (actividad, destino))
relaciones = {
    1: [('A', 2), ('C', 3)],  # Nodo 1 se conecta con 2 (A) y 3 (C)
    2: [('F1', 3), ('B', 4)], # Nodo 2 se conecta con 3 (F1) y 4 (B)
    3: [('D', 4)]             # Nodo 3 se conecta con 4 (D)
}

# Duraciones de las actividades (t0, tm, tp calculado previamente como De)
duraciones = {
    'A': 89 / 6,
    'B': 55 / 6,
    'C': 51 / 6,
    'D': 41 / 6,
    'F1': 0
}

# Calcular Early Times (Ei)
Ei, tabla_early = calcular_early_times(nodos, relaciones, duraciones)

# Calcular Late Times (Li)
Li, tabla_late = calcular_late_times(nodos, relaciones, duraciones, Ei)

# Unir las dos tablas en una sola
tabla_completa = pd.merge(tabla_early, tabla_late, on='Ti')

# Generar la tabla de tareas
tabla_tareas = generar_tabla_tareas(nodos, relaciones, duraciones, Ei, Li)

# Mostrar las tablas generadas
print("Tabla Completa (Ei y Li):")
print(tabla_completa.to_latex(index=False, float_format="%.3f"))

print("\nTabla de Tareas:")
print(tabla_tareas.to_latex(index=False, float_format="%.3f"))

# Filtrar las tareas críticas
tareas_criticas = tabla_tareas[tabla_tareas['Critico'] == 'Sí']

# Mostrar el camino crítico
camino_critico = tareas_criticas[['Tarea', 'Ruta(i->j)', 'Di']]

# Calcular la suma de las duraciones de las tareas críticas
suma_criticas = tareas_criticas['Di'].sum()

# Mostrar el camino crítico y la suma de las duraciones
print("\nCamino Crítico:")
print(camino_critico.to_latex(index=False, float_format="%.3f"))

print(f"\nSuma de las duraciones de las tareas críticas: {suma_criticas:.3f}")

# Exportar las tablas a LaTeX (opcional)
archivo_latex_completo = "tabla_completa.tex"
tabla_completa.to_latex(archivo_latex_completo, index=False, float_format="%.3f")

archivo_latex_tareas = "tabla_tareas.tex"
tabla_tareas.to_latex(archivo_latex_tareas, index=False, float_format="%.3f")

print(f"\nLa tabla completa se ha exportado a: {archivo_latex_completo}")
print(f"La tabla de tareas se ha exportado a: {archivo_latex_tareas}")
